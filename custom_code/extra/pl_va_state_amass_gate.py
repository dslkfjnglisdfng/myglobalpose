"""Gate a temporally consistent AMASS cache before formal PL-VA training."""

import argparse
import json
from pathlib import Path

import torch

from l4_train_diverse_short import load_cache_files
from pl_va_state import causal_world_angular_velocity_from_rmb_sequence
from pl_va_state_frame_audit import angular_step_quantiles, centered_world_omega, stats


def run(manifest, output, sequences_per_shard=10, pearson_min=0.8):
    files, _ = load_cache_files(manifest)
    pred, ref, rows = [], [], []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=False)
        for i in range(min(sequences_per_shard, len(data["name"]))):
            rmb = data["RMB"][i].float()
            new = causal_world_angular_velocity_from_rmb_sequence(rmb)
            fk_rotation_ref = centered_world_omega(rmb)
            pred.append(new[2:]); ref.append(fk_rotation_ref[2:])
            rows.append({"sequence": str(data["name"][i]),
                         "view": str(data.get("view_type", [""] * len(data["name"]))[i]),
                         **angular_step_quantiles(rmb)})
    metric = stats(torch.cat(pred), torch.cat(ref))
    original = [x for x in rows if x["view"] != "offset_aug_overlay"]
    overlay = [x for x in rows if x["view"] == "offset_aug_overlay"]
    def aggregate(group):
        return {key: float(torch.tensor([x[key] for x in group]).median()) for key in
                ("p50_rad", "p95_rad", "p99_rad", "max_rad")} if group else {}
    jitter_ok = bool(rows) and aggregate(original).get("p50_rad", 1.0) < 0.05
    passed = metric["pearson"] >= pearson_min and jitter_ok
    result = {"status": "passed" if passed else "failed", "formal_training_allowed": passed,
              "manifest": str(manifest), "sampled_sequences": len(rows),
              "reference": "GT pose FK global IMU-joint rotations; centered SO3 derivative for offline gate only",
              "new_lag2_ema03_vs_fk_rotation": metric, "pearson_min": pearson_min,
              "no_independent_rmb_jitter_dominance": jitter_ok,
              "angular_step_median_of_sequence_quantiles": {"original": aggregate(original), "offset_overlay": aggregate(overlay)},
              "per_sequence": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequences-per-shard", type=int, default=10)
    p.add_argument("--pearson-min", type=float, default=0.8)
    a = p.parse_args()
    result = run(a.manifest, a.output, a.sequences_per_shard, a.pearson_min)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sequence"}, indent=2))
    raise SystemExit(0 if result["formal_training_allowed"] else 2)


if __name__ == "__main__":
    main()
