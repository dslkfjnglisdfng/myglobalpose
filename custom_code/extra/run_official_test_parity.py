"""Run the unchanged official test.compare_realimu flow for parity/G2."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def _load_official_modules(repo_root: Path):
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    net = importlib.import_module("net")
    official_test = importlib.import_module("test")
    official_test.plt.show = lambda: None
    return net, official_test


def _select_data(data, dataset: str):
    if dataset == "dip":
        return data
    indices = [i for i, name in enumerate(data["name"]) if str(name).startswith("s5_")]
    if [data["name"][i] for i in indices] != [
        "s5_freestyle1", "s5_freestyle3", "s5_rom3", "s5_walking2"
    ]:
        raise RuntimeError("TotalCapture official s5 split does not match the required four sequences")
    return {key: [value[i] for i in indices] if isinstance(value, list) else value for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline_original", "current_g0", "g2_vr_swap"), required=True)
    parser.add_argument("--dataset", choices=("dip", "totalcapture"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    net, official_test = _load_official_modules(repo_root)

    if args.variant == "g2_vr_swap":
        extra = repo_root / "custom_code" / "extra"
        sys.path.insert(0, str(extra))
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

    dataset_file = "dipimu.pt" if args.dataset == "dip" else "totalcapture_officalib.pt"
    dataset_name = "DIP-IMU" if args.dataset == "dip" else "TotalCapture-s5-Official"
    data = torch.load(repo_root / "data" / "test_datasets" / dataset_file)
    data = _select_data(data, args.dataset)
    if args.max_sequences:
        data = {key: value[:args.max_sequences] if isinstance(value, list) else value for key, value in data.items()}
    if args.max_frames:
        data = {
            key: [sequence[:args.max_frames] for sequence in value]
            if isinstance(value, list) and key != "name" else value
            for key, value in data.items()
        }
    official_test.compare_realimu(
        data,
        dataset_name=dataset_name,
        save_results_dir=str(output_dir),
        evaluate_pose=True,
        evaluate_tran=args.dataset == "totalcapture",
    )
    generated = output_dir / f"{dataset_name}_GlobalPose.pt"
    if not generated.exists():
        raise FileNotFoundError(generated)
    shutil.move(generated, output_dir / "predictions.pt")


if __name__ == "__main__":
    main()
