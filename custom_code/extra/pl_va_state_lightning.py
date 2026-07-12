"""PyTorch Lightning training logic for PL-VA-State-V1.

Read this file from top to bottom: loss construction, one training/validation
step, optimizer creation, then checkpoint compatibility.  The network and the
p/v/a state equations remain in :mod:`pl_va_state`; this module only trains it.
"""

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_va_state import PLVAStateV1


LOSS_WEIGHTS = {
    "p": 1.0,
    "v_direct": 0.25,
    "v_state": 0.25,
    "a": 0.1,
    "g": 1.0,
    "consistency": 0.2,
    "jerk": 0.01,
}


def masked_huber(prediction, target, mask):
    """SmoothL1 averaged over vector channels, then over valid frames only."""
    per_frame = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    return per_frame[mask].mean()


def compute_losses(output, batch, stats):
    """Return total, raw, and weighted losses under the original PL-VA recipe."""
    mask = batch["mask"]
    velocity_direct = masked_huber(
        (output["vRB_direct"] - stats["v_mean"]) / stats["v_std"],
        (batch["v"] - stats["v_mean"]) / stats["v_std"],
        mask,
    )
    velocity_state = masked_huber(
        (output["vRB_state"] - stats["v_mean"]) / stats["v_std"],
        (batch["v"] - stats["v_mean"]) / stats["v_std"],
        mask,
    )
    acceleration = masked_huber(
        (output["aRB_leaf"] - stats["a_mean"]) / stats["a_std"],
        (batch["a"] - stats["a_mean"]) / stats["a_std"],
        mask,
    )
    raw = {
        "p": masked_huber(output["pRB_state"], batch["p"], mask),
        "v_direct": velocity_direct,
        "v_state": velocity_state,
        "a": acceleration,
        "g": (1 - (output["gR1"] * batch["g"]).sum(-1).clamp(-1, 1))[mask].mean(),
    }

    # Pairwise losses require both neighboring frames to be valid.
    pair_mask = mask[:, 1:] & mask[:, :-1]
    delta_velocity = output["vRB_state"][:, 1:] - output["vRB_state"][:, :-1]
    acceleration_impulse = 0.5 / 60 * (
        output["aRB_leaf"][:, :-1] + output["aRB_leaf"][:, 1:]
    )
    raw["consistency"] = masked_huber(
        delta_velocity, acceleration_impulse, pair_mask
    )
    raw["jerk"] = masked_huber(
        output["aRB_leaf"][:, 1:] - output["aRB_leaf"][:, :-1],
        torch.zeros_like(acceleration_impulse),
        pair_mask,
    )
    weighted = {name: raw[name] * LOSS_WEIGHTS[name] for name in raw}
    return sum(weighted.values()), raw, weighted


def selection_metric(raw):
    """The physically meaningful metric used to select the best checkpoint."""
    return raw["p"] + 0.25 * raw["v_state"] + 0.1 * raw["a"] + raw["g"]


class PLVAStateLightning(pl.LightningModule):
    """Train ``PLVAStateV1`` while exposing readable Lightning lifecycle hooks."""

    def __init__(self, stats, learning_rate=1e-4, epoch_offset=0,
                 legacy_optimizer_state=None):
        super().__init__()
        checkpoint_stats = {name: value.detach().cpu()
                            for name, value in stats.items()
                            if torch.is_tensor(value)}
        self.save_hyperparameters(
            {"learning_rate": learning_rate, "epoch_offset": epoch_offset,
             "stats": checkpoint_stats},
        )
        self.model = PLVAStateV1()
        self.register_buffer("v_mean", stats["v_mean"].clone())
        self.register_buffer("v_std", stats["v_std"].clone())
        self.register_buffer("a_mean", stats["a_mean"].clone())
        self.register_buffer("a_std", stats["a_std"].clone())
        self.legacy_optimizer_state = legacy_optimizer_state
        self.gradient_norms = {}

    @property
    def stats(self):
        return {"v_mean": self.v_mean, "v_std": self.v_std,
                "a_mean": self.a_mean, "a_std": self.a_std,
                "fit_split": "train_only"}

    def forward(self, batch):
        return self.model.forward_sequence(
            batch["feature"], batch["init"], batch["lengths"]
        )

    def _shared_step(self, batch, stage):
        output = self(batch)
        total, raw, weighted = compute_losses(output, batch, self.stats)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite {stage} loss")

        batch_size = batch["feature"].shape[0]
        self.log(f"{stage}/total", total, on_step=False, on_epoch=True,
                 batch_size=batch_size, prog_bar=(stage == "val"))
        for name, value in raw.items():
            self.log(f"{stage}/raw_{name}", value, on_step=False, on_epoch=True,
                     batch_size=batch_size)
            self.log(f"{stage}/weighted_{name}", weighted[name],
                     on_step=False, on_epoch=True, batch_size=batch_size)
        if stage == "val":
            self.log("val/selection", selection_metric(raw), on_step=False,
                     on_epoch=True, batch_size=batch_size, prog_bar=True)
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def on_fit_start(self):
        """Import optimizer moments once when migrating a legacy ``last.pt``."""
        if self.legacy_optimizer_state is not None:
            self.trainer.optimizers[0].load_state_dict(self.legacy_optimizer_state)
            self.legacy_optimizer_state = None

    def on_after_backward(self):
        """Record the three output-head gradient norms for smoke diagnostics."""
        gradient = self.model.net.linear2.weight.grad
        if gradient is not None:
            self.gradient_norms = {
                "v_head": float(gradient[:15].norm()),
                "a_head": float(gradient[15:30].norm()),
                "gravity_head": float(gradient[30:33].norm()),
            }


class ProjectCheckpointCallback(pl.Callback):
    """Save Lightning checkpoints and the legacy files consumed by evaluators."""

    def __init__(self, output_dir, config, initial_best=float("inf")):
        self.output_dir = Path(output_dir)
        self.config = config
        self.best_selection = initial_best
        self.last_validation = {}
        self.last_epoch = 0

    def on_validation_epoch_end(self, trainer, module):
        metrics = trainer.callback_metrics
        selection = float(metrics["val/selection"])
        self.last_validation = {
            "p": float(metrics["val/raw_p"]),
            "v_state": float(metrics["val/raw_v_state"]),
            "a": float(metrics["val/raw_a"]),
            "g": float(metrics["val/raw_g"]),
            "total": float(metrics["val/total"]),
            "selection": selection,
        }
        epoch = module.hparams.epoch_offset + trainer.current_epoch + 1
        self.last_epoch = epoch
        state = self._legacy_state(trainer, module, epoch)

        torch.save(state, self.output_dir / "last.pt")
        trainer.save_checkpoint(self.output_dir / "last.ckpt")
        if selection < self.best_selection:
            self.best_selection = selection
            torch.save(state, self.output_dir / "best.pt")
            trainer.save_checkpoint(self.output_dir / "best.ckpt")

        row = {"epoch": epoch, "validation": self.last_validation,
               "gradient_norm": module.gradient_norms}
        print(json.dumps(row), flush=True)

    def on_fit_end(self, trainer, module):
        summary = {
            "status": "ok",
            "finite": True,
            "epochs": self.last_epoch,
            "resumed_from_epoch": module.hparams.epoch_offset,
            "best_selection": self.best_selection,
            "last": {"validation": self.last_validation},
            "gradient_norm": module.gradient_norms,
            "trainer": "pytorch_lightning",
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def _legacy_state(self, trainer, module, epoch):
        return {
            "model": module.model.state_dict(),
            "optimizer": trainer.optimizers[0].state_dict(),
            "epoch": epoch,
            "normalization": {name: value.detach().cpu()
                              for name, value in module.stats.items()
                              if torch.is_tensor(value)},
            "config": self.config,
            "validation": self.last_validation,
        }
