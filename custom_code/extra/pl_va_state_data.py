"""Data loading for PL-VA-State-V1.

The state integrator must see each motion from its real first frame to its real
last frame.  Therefore one dataset item is one complete sequence.  Batches are
formed from similarly sized sequences to reduce padding without cutting or
shuffling time inside a sequence.
"""

import json
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pl_va_state import ANGULAR_VELOCITY_METHOD


def load_records(manifest_path, max_sequences=0):
    """Load complete cached sequences and verify the angular-input contract."""
    manifest = json.loads(Path(manifest_path).read_text())
    angular = manifest.get("angular_velocity", {})
    if angular.get("method") != ANGULAR_VELOCITY_METHOD:
        raise ValueError(
            "refusing incompatible PL-VA cache angular velocity: "
            f"{angular.get('method')!r}"
        )

    records = []
    for item in manifest["cache_files"]:
        records.extend(torch.load(item["path"], map_location="cpu"))
        if max_sequences and len(records) >= max_sequences:
            break
    return records[:max_sequences or None], manifest


def normalization(records):
    """Fit velocity/acceleration statistics from the training split only."""
    velocity = torch.cat([record["v_gt"] for record in records])
    acceleration = torch.cat([record["a_gt"] for record in records])
    return {
        "v_mean": velocity.mean(0),
        "v_std": velocity.std(0).clamp_min(1e-4),
        "a_mean": acceleration.mean(0),
        "a_std": acceleration.std(0).clamp_min(1e-4),
        "fit_split": "train_only",
    }


class PLVASequenceDataset(Dataset):
    """Thin dataset: cached records already contain all model-ready tensors."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class LengthBucketBatchSampler(Sampler):
    """Group neighboring sequence lengths, then shuffle whole batch groups.

    Sorting cuts padding substantially.  Shuffling groups preserves stochastic
    batch order while never changing frame order or resetting sequence state.
    """

    def __init__(self, lengths, batch_size, shuffle, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = sorted(range(len(lengths)), key=lambda index: lengths[index])
        self.groups = [order[start:start + batch_size]
                       for start in range(0, len(order), batch_size)]

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
            self.epoch += 1
        yield from groups

    def __len__(self):
        return len(self.groups)


def collate_sequences(records):
    """Pad complete sequences and return a mask for every supervised tensor."""
    lengths = torch.tensor([record["length"] for record in records], dtype=torch.long)
    max_length = int(lengths.max())

    def pad(key, width):
        output = torch.zeros(len(records), max_length, width)
        for index, record in enumerate(records):
            output[index, :record["length"]] = record[key]
        return output

    mask = torch.arange(max_length)[None] < lengths[:, None]
    return {
        "feature": pad("feature", 102),
        "p": pad("p_gt", 15),
        "v": pad("v_gt", 15),
        "a": pad("a_gt", 15),
        "g": pad("g_gt", 3),
        "init": torch.stack([record["init_legacy"] for record in records]),
        "mask": mask,
        "lengths": lengths,
    }


class PLVADataModule(pl.LightningDataModule):
    """Own the cache split and the two complete-sequence DataLoaders."""

    def __init__(self, train_cache, val_cache, batch_size=8,
                 max_train_sequences=0, max_val_sequences=0, seed=42):
        super().__init__()
        self.train_cache = Path(train_cache)
        self.val_cache = Path(val_cache)
        self.batch_size = batch_size
        self.max_train_sequences = max_train_sequences
        self.max_val_sequences = max_val_sequences
        self.seed = seed
        self.train_records = None
        self.val_records = None
        self.train_manifest = None

    def setup(self, stage=None):
        if self.train_records is not None:
            return
        self.train_records, self.train_manifest = load_records(
            self.train_cache, self.max_train_sequences
        )
        self.val_records, _ = load_records(self.val_cache, self.max_val_sequences)

    def training_normalization(self):
        self.setup("fit")
        return normalization(self.train_records)

    def _loader(self, records, shuffle):
        dataset = PLVASequenceDataset(records)
        sampler = LengthBucketBatchSampler(
            [record["length"] for record in records],
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sequences)

    def train_dataloader(self):
        return self._loader(self.train_records, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_records, shuffle=False)

    def summary(self):
        self.setup("fit")
        return {
            "train_sequences": len(self.train_records),
            "val_sequences": len(self.val_records),
            "train_batches": len(self.train_dataloader()),
            "val_batches": len(self.val_dataloader()),
            "batch_size": self.batch_size,
            "sampling": "complete_sequences_length_bucketed",
        }
