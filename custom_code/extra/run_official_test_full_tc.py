"""Run unchanged test.compare_realimu on every sequence in a TC release cache."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


DATASETS = {
    "officalib": ("totalcapture_officalib.pt", "TotalCapture (Official Calibration)"),
    "dipcalib": ("totalcapture_dipcalib.pt", "TotalCapture (DIP Calibration)"),
}


def deterministic_setup() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_official_modules(repo_root: Path):
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    net = importlib.import_module("net")
    official_test = importlib.import_module("test")
    official_test.plt.show = lambda: None
    return net, official_test


def inventory(data: dict, dataset_path: Path) -> dict:
    names = data.get("name")
    rows = []
    for i, pose in enumerate(data["pose"]):
        rows.append({
            "index": i,
            "name": str(names[i]) if names is not None else None,
            "frames": int(pose.shape[0]),
        })
    return {
        "path": str(dataset_path.resolve()),
        "sequence_count": len(rows),
        "total_frames": sum(row["frames"] for row in rows),
        "sequences": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline_original", "current_g0", "g2_vr_swap"), required=True)
    parser.add_argument("--calibration", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    deterministic_setup()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    net, official_test = load_official_modules(repo_root)

    if args.variant == "g2_vr_swap":
        sys.path.insert(0, str(repo_root / "custom_code" / "extra"))
        from gp_w_input_swap import CausalRMBWorldAngularVelocity

        class G2VRNet(net.GPNet):
            def __init__(self):
                super().__init__()
                self._causal_w = CausalRMBWorldAngularVelocity()

            def rnn_initialize(self, *init_args, **init_kwargs):
                result = super().rnn_initialize(*init_args, **init_kwargs)
                self._causal_w.reset()
                return result

            def forward_frame(self, a, w, rmb):
                w_vr = self._causal_w.step(rmb)
                return super().forward_frame(a, w, rmb, w_vr_override=w_vr)

        official_test.GPNet = G2VRNet

    filename, dataset_name = DATASETS[args.calibration]
    dataset_path = repo_root / "data" / "test_datasets" / filename
    data = torch.load(dataset_path)
    dataset_inventory = inventory(data, dataset_path)
    print(json.dumps(dataset_inventory, indent=2), flush=True)

    # Deliberately pass the complete release object unchanged. compare_realimu
    # owns the official for seq_idx in range(len(data['pose'])) traversal.
    official_test.compare_realimu(
        data,
        dataset_name=dataset_name,
        save_results_dir=str(output_dir),
        evaluate_pose=True,
        evaluate_tran=True,
    )
    generated = output_dir / f"{dataset_name}_GlobalPose.pt"
    if not generated.exists():
        raise FileNotFoundError(generated)
    shutil.move(generated, output_dir / "predictions.pt")
    (output_dir / "dataset_inventory.json").write_text(json.dumps(dataset_inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
