"""Readable command-line entry point for PL-VA-State-V1 training.

Reading order:
1. :mod:`pl_va_state_data` explains complete-sequence batching.
2. :mod:`pl_va_state_lightning` explains losses and training steps.
3. This file only validates arguments, initializes weights, and starts Trainer.
"""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from pl_va_state import (
    ANGULAR_VELOCITY_EMA_BETA,
    ANGULAR_VELOCITY_FRAME,
    ANGULAR_VELOCITY_LAG,
    ANGULAR_VELOCITY_METHOD,
    partial_initialize_from_official,
)
from pl_va_state_data import PLVADataModule, load_records
from pl_va_state_lightning import (
    LOSS_WEIGHTS,
    PLVAStateLightning,
    ProjectCheckpointCallback,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("data/weights.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def load_legacy_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    method = checkpoint.get("config", {}).get("angular_velocity_method")
    if method != ANGULAR_VELOCITY_METHOD:
        raise ValueError("refusing checkpoint with an incompatible angular feature")
    return checkpoint


def historical_best(output_dir, fallback=float("inf")):
    path = output_dir / "best.pt"
    if not path.exists():
        return fallback
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return float(checkpoint.get("validation", {}).get("selection", fallback))


def build_config(args, data_summary):
    return {
        **vars(args),
        "loss_weights": LOSS_WEIGHTS,
        "state_reset": "once_per_full_sequence",
        "trainer": "pytorch_lightning_2.6",
        "data": data_summary,
        "angular_velocity_method": ANGULAR_VELOCITY_METHOD,
        "angular_velocity_frame": ANGULAR_VELOCITY_FRAME,
        "angular_velocity_lag": ANGULAR_VELOCITY_LAG,
        "angular_velocity_ema_beta": ANGULAR_VELOCITY_EMA_BETA,
    }


def main():
    args = parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("use only one of --init-checkpoint and --resume-checkpoint")

    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = PLVADataModule(
        args.train_cache,
        args.val_cache,
        batch_size=args.batch_size,
        max_train_sequences=args.max_train_sequences,
        max_val_sequences=args.max_val_sequences,
        seed=args.seed,
    )
    data.setup("fit")

    start_epoch = 0
    epoch_offset = 0
    optimizer_state = None
    stats = data.training_normalization()
    legacy_checkpoint = None
    lightning_checkpoint = None
    if args.resume_checkpoint and args.resume_checkpoint.suffix == ".ckpt":
        lightning_checkpoint = args.resume_checkpoint
        module = PLVAStateLightning.load_from_checkpoint(lightning_checkpoint)
        epoch_offset = int(module.hparams.epoch_offset)
        native_state = torch.load(lightning_checkpoint, map_location="cpu", weights_only=False)
        start_epoch = epoch_offset + int(native_state["epoch"]) + 1
    elif args.init_checkpoint or args.resume_checkpoint:
        legacy_checkpoint = load_legacy_checkpoint(
            args.resume_checkpoint or args.init_checkpoint
        )
        stats = legacy_checkpoint["normalization"]
        if args.resume_checkpoint:
            start_epoch = int(legacy_checkpoint.get("epoch", 0))
            epoch_offset = start_epoch
            optimizer_state = legacy_checkpoint.get("optimizer")

    if lightning_checkpoint is None:
        module = PLVAStateLightning(
            stats=stats,
            learning_rate=args.lr,
            epoch_offset=epoch_offset,
            legacy_optimizer_state=optimizer_state,
        )
        if legacy_checkpoint is not None:
            module.model.load_state_dict(legacy_checkpoint["model"])
        else:
            partial_initialize_from_official(
                module.model, args.weights,
                args.output_dir / "initialization_report.json",
            )

    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        print(json.dumps({"status": "already_complete", "epoch": start_epoch}))
        return

    config = build_config(args, data.summary())
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    checkpoint_callback = ProjectCheckpointCallback(
        args.output_dir,
        config,
        initial_best=historical_best(args.output_dir),
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # Lightning counts epochs inside its current run.  ``epoch_offset``
        # maps those local epochs back to the project-wide AMASS/DIP count.
        max_epochs=args.epochs - epoch_offset,
        gradient_clip_val=args.grad_clip,
        logger=CSVLogger(args.output_dir, name="lightning_logs"),
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=data, ckpt_path=lightning_checkpoint)


if __name__ == "__main__":
    main()
