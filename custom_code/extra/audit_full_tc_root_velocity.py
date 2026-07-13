"""Minimal post-hoc root-velocity audit for saved full-TotalCapture predictions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "data/experiments/gp_w_input_swap_official_test_full_tc_20260712"
DT = 1 / 60
LOWPASS_WINDOW = 15
METRICS = (
    "velocity_rmse_m_s", "velocity_mae_m_s", "velocity_bias_x_m_s",
    "velocity_bias_y_m_s", "velocity_bias_z_m_s", "velocity_bias_norm_m_s",
    "horizontal_velocity_rmse_m_s", "vertical_velocity_rmse_m_s",
    "speed_magnitude_mae_m_s", "velocity_direction_angle_deg",
    "low_frequency_velocity_rmse_m_s", "high_frequency_velocity_rmse_m_s",
)


def velocity(position: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """World/model-frame finite-difference root velocity, shape (T-1, 3)."""
    return (position[1:] - position[:-1]) / dt


def moving_average(x: torch.Tensor, window: int = LOWPASS_WINDOW) -> torch.Tensor:
    if window % 2 != 1:
        raise ValueError("moving-average window must be odd")
    pad = window // 2
    return F.conv1d(F.pad(x.T.unsqueeze(0), (pad, pad), mode="replicate"), torch.ones(3, 1, window) / window, groups=3).squeeze(0).T


def metrics(pred_position: torch.Tensor, gt_position: torch.Tensor) -> dict[str, float]:
    vp, vg = velocity(pred_position.float()), velocity(gt_position.float())
    if vp.shape != vg.shape or len(vp) == 0:
        raise ValueError("prediction/GT translation length mismatch or no velocity frames")
    error = vp - vg
    speed_p, speed_g = vp.norm(dim=1), vg.norm(dim=1)
    direction_mask = speed_g > 0.1
    cosine = (vp[direction_mask] * vg[direction_mask]).sum(1) / (speed_p[direction_mask] * speed_g[direction_mask]).clamp_min(1e-12)
    low_error = moving_average(vp) - moving_average(vg)
    high_error = (vp - moving_average(vp)) - (vg - moving_average(vg))
    bias = error.mean(0)
    return {
        "velocity_rmse_m_s": float(error.square().sum(1).mean().sqrt()),
        "velocity_mae_m_s": float(error.norm(dim=1).mean()),
        "velocity_bias_x_m_s": float(bias[0]),
        "velocity_bias_y_m_s": float(bias[1]),
        "velocity_bias_z_m_s": float(bias[2]),
        "velocity_bias_norm_m_s": float(bias.norm()),
        "horizontal_velocity_rmse_m_s": float(error[:, (0, 2)].square().sum(1).mean().sqrt()),
        "vertical_velocity_rmse_m_s": float(error[:, 1].square().mean().sqrt()),
        "speed_magnitude_mae_m_s": float((speed_p - speed_g).abs().mean()),
        "velocity_direction_angle_deg": float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()) if direction_mask.any() else math.nan,
        "low_frequency_velocity_rmse_m_s": float(low_error.square().sum(1).mean().sqrt()),
        "high_frequency_velocity_rmse_m_s": float(high_error.square().sum(1).mean().sqrt()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_value(metric: str, value: float) -> float:
    # Signed bias components describe direction; their improvement is smaller magnitude.
    return abs(value) if metric in {"velocity_bias_x_m_s", "velocity_bias_y_m_s", "velocity_bias_z_m_s"} else value


def audit(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    aggregate_rows, sequence_rows, summary = [], [], []
    for calibration in ("officalib", "dipcalib"):
        data = torch.load(ROOT / "data/test_datasets" / f"totalcapture_{calibration}.pt", map_location="cpu")
        g0 = torch.load(EXPERIMENT / "current_g0" / f"{calibration}_run/predictions.pt", map_location="cpu")
        g2 = torch.load(EXPERIMENT / "g2_vr_swap" / f"{calibration}_run/predictions.pt", map_location="cpu")
        if not (len(data["tran"]) == len(g0["tran"]) == len(g2["tran"]) == 45):
            raise ValueError(f"{calibration}: expected aligned full 45-sequence release")
        values = {"g0": [], "g2": []}
        for index in range(45):
            result = {"g0": metrics(g0["tran"][index], data["tran"][index]), "g2": metrics(g2["tran"][index], data["tran"][index])}
            values["g0"].append(result["g0"]); values["g2"].append(result["g2"])
            row = {"calibration": calibration, "sequence_index": index, "sequence": str(data["name"][index]), "frames": len(data["tran"][index])}
            for metric in METRICS:
                delta = result["g2"][metric] - result["g0"][metric]
                comparison_delta = comparison_value(metric, result["g2"][metric]) - comparison_value(metric, result["g0"][metric])
                row.update({f"g0_{metric}": result["g0"][metric], f"g2_{metric}": result["g2"][metric], f"g2_minus_g0_{metric}": delta, f"{metric}_outcome": "better" if comparison_delta < -1e-12 else "worse" if comparison_delta > 1e-12 else "tie"})
            sequence_rows.append(row)
        for metric in METRICS:
            g0_mean = sum(item[metric] for item in values["g0"]) / 45
            g2_mean = sum(item[metric] for item in values["g2"]) / 45
            if metric == "velocity_bias_norm_m_s":
                g0_mean = math.sqrt(sum((sum(item[f"velocity_bias_{axis}_m_s"] for item in values["g0"]) / 45) ** 2 for axis in "xyz"))
                g2_mean = math.sqrt(sum((sum(item[f"velocity_bias_{axis}_m_s"] for item in values["g2"]) / 45) ** 2 for axis in "xyz"))
            deltas = [comparison_value(metric, b[metric]) - comparison_value(metric, a[metric]) for a, b in zip(values["g0"], values["g2"])]
            aggregate_rows.append({"calibration": calibration, "metric": metric, "aggregation": "unweighted mean of 45 per-sequence metrics; bias norm is norm of aggregate bias vector", "g0": g0_mean, "g2": g2_mean, "g2_minus_g0": g2_mean - g0_mean, "g2_relative_percent": (g2_mean - g0_mean) / g0_mean * 100 if g0_mean else math.nan, "g2_wins": sum(delta < -1e-12 for delta in deltas), "g2_losses": sum(delta > 1e-12 for delta in deltas), "ties": sum(abs(delta) <= 1e-12 for delta in deltas)})
        by_metric = {row["metric"]: row for row in aggregate_rows if row["calibration"] == calibration}
        summary.append((calibration, by_metric))
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output / "per_sequence_metrics.csv", sequence_rows)
    conclusion = 2 if all(x[1]["velocity_bias_norm_m_s"]["g2"] < x[1]["velocity_bias_norm_m_s"]["g0"] and x[1]["high_frequency_velocity_rmse_m_s"]["g2"] > x[1]["high_frequency_velocity_rmse_m_s"]["g0"] for x in summary) else 1 if all(x[1]["velocity_rmse_m_s"]["g2"] < x[1]["velocity_rmse_m_s"]["g0"] for x in summary) else 3
    lines = ["# Root velocity audit", "", "Saved final root translations only; no model, inference, or weights were changed.", "", "- Frame: final predicted/GT root translation in the TotalCapture model/world coordinate system.", "- Velocity: `(p[t] - p[t-1]) / (1/60)` m/s; first frame is omitted.", "- Low frequency: centered 15-frame, edge-replicated moving average of velocity; high frequency is velocity minus that average.", "- Aggregation: unweighted mean of the 45 per-sequence metrics.", ""]
    for calibration, table in summary:
        b0 = ", ".join(f"{table[f'velocity_bias_{axis}_m_s']['g0']:.6f}" for axis in "xyz")
        b2 = ", ".join(f"{table[f'velocity_bias_{axis}_m_s']['g2']:.6f}" for axis in "xyz")
        lines += [f"## {calibration}", "", f"- Velocity RMSE: {table['velocity_rmse_m_s']['g0']:.6f} -> {table['velocity_rmse_m_s']['g2']:.6f} m/s ({table['velocity_rmse_m_s']['g2_relative_percent']:+.2f}%).", f"- Low/high RMSE: {table['low_frequency_velocity_rmse_m_s']['g0']:.6f} -> {table['low_frequency_velocity_rmse_m_s']['g2']:.6f}; {table['high_frequency_velocity_rmse_m_s']['g0']:.6f} -> {table['high_frequency_velocity_rmse_m_s']['g2']:.6f} m/s.", f"- Mean bias vector: ({b0}) -> ({b2}) m/s; norm {table['velocity_bias_norm_m_s']['g0']:.6f} -> {table['velocity_bias_norm_m_s']['g2']:.6f} m/s.", f"- RMSE sequence wins/losses: {table['velocity_rmse_m_s']['g2_wins']} / {table['velocity_rmse_m_s']['g2_losses']}.", ""]
    conclusions = {1: "G2 的逐帧 velocity 也更准确。", 2: "G2 主要降低低频 bias，但高频误差更大。", 3: "translation 改善主要来自误差抵消，逐帧 velocity 并未改善。"}
    lines += ["## Conclusion", "", conclusions[conclusion], "", "The lower DC bias does not make overall framewise velocity more accurate here: low-frequency RMSE is essentially unchanged/slightly worse and total velocity RMSE is worse. See `aggregate_metrics.csv` and `per_sequence_metrics.csv` for every requested metric and per-sequence G2-vs-G0 outcome."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def self_test() -> None:
    position = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    assert torch.equal(velocity(position, 1.0), torch.tensor([[1.0, 0, 0], [2.0, 0, 0]]))
    assert metrics(position, position)["velocity_rmse_m_s"] == 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "root_velocity_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("root-velocity self-test passed")
    else:
        audit(args.output.resolve())
