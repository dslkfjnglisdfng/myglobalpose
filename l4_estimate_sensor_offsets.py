import argparse
from pathlib import Path

import torch

from l4_sensor_offset_utils import (
    FPS,
    SENSOR_NAMES,
    make_metadata,
    prepare_sequence_kinematics,
    theoretical_vertex_offsets,
    load_dataset_file,
)


def robust_weights(residual, delta):
    norm = residual.norm(dim=-1).clamp_min(1e-12)
    return torch.clamp(torch.as_tensor(delta, dtype=residual.dtype) / norm, max=1.0)


def solve_sensor_offset(
    ddot_p_wj,
    ddot_R_wj,
    a_obs,
    start,
    end,
    ridge=1e-4,
    fit_bias=False,
    huber_delta=0.0,
    irls_iters=5,
):
    c = ddot_p_wj[start:end].float()
    A = ddot_R_wj[start:end].float().reshape(end - start, 3, 3)
    y = a_obs[start:end].float() - c
    valid = torch.isfinite(A).all(dim=(1, 2)) & torch.isfinite(y).all(dim=1)
    if valid.sum() < 6:
        return None
    c_valid = c[valid]
    obs_valid = a_obs[start:end].float()[valid]
    A = A[valid]
    y = y[valid]
    n = A.shape[0]
    if fit_bias:
        M = torch.cat((A, -torch.eye(3).view(1, 3, 3).expand(n, -1, -1)), dim=-1).reshape(-1, 6)
        reg = torch.diag(torch.tensor([ridge, ridge, ridge, ridge, ridge, ridge], dtype=M.dtype))
    else:
        M = A.reshape(-1, 3)
        reg = torch.eye(3, dtype=M.dtype) * ridge
    target = y.reshape(-1)

    weights = torch.ones(n, dtype=M.dtype)
    x = None
    for _ in range(max(1, int(irls_iters))):
        row_weights = weights.repeat_interleave(3).sqrt()
        Mw = M * row_weights[:, None]
        yw = target * row_weights
        lhs = Mw.T.matmul(Mw) + reg
        rhs = Mw.T.matmul(yw)
        try:
            x = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            x = torch.linalg.lstsq(lhs, rhs).solution
        if huber_delta <= 0:
            break
        pred = M.matmul(x).view(-1, 3)
        residual = pred - y
        weights = robust_weights(residual, huber_delta)

    pred_zero = c_valid
    if fit_bias:
        r = x[:3]
        bias = x[3:]
        pred_fit = c_valid + A.matmul(r.view(3, 1)).squeeze(-1) - bias.view(1, 3)
    else:
        r = x
        bias = torch.zeros(3, dtype=x.dtype)
        pred_fit = c_valid + A.matmul(r.view(3, 1)).squeeze(-1)
    residual_zero = (pred_zero - obs_valid).norm(dim=-1).mean()
    residual_fit = (pred_fit - obs_valid).norm(dim=-1).mean()
    improvement = (residual_zero - residual_fit) / residual_zero.clamp_min(1e-12)
    s = torch.linalg.svdvals(A.reshape(-1, 3))
    smax = s.max() if s.numel() else torch.tensor(float("nan"))
    smin = s.min() if s.numel() else torch.tensor(float("nan"))
    cond = smax / smin.clamp_min(1e-12)
    observability = smin
    return {
        "offset": r.float(),
        "bias": bias.float(),
        "residual_zero": residual_zero.float(),
        "residual_fit": residual_fit.float(),
        "residual_improvement": improvement.float(),
        "condition_number": cond.float(),
        "singular_values": s.float(),
        "observability_score": observability.float(),
        "num_valid_frames": int(valid.sum().item()),
    }


def window_ranges(num_frames, window_size, stride):
    if window_size <= 0 or window_size >= num_frames:
        return [(0, num_frames)]
    ranges = []
    start = 0
    while start + window_size <= num_frames:
        ranges.append((start, start + window_size))
        start += stride
    if ranges and ranges[-1][1] < num_frames:
        ranges.append((num_frames - window_size, num_frames))
    return ranges or [(0, num_frames)]


def aggregate_window_results(results, max_offset_norm, max_condition, min_improvement):
    good = []
    for result in results:
        if result is None:
            continue
        norm = result["offset"].norm()
        is_good = (
            torch.isfinite(norm)
            and norm <= max_offset_norm
            and torch.isfinite(result["condition_number"])
            and result["condition_number"] <= max_condition
            and torch.isfinite(result["residual_improvement"])
            and result["residual_improvement"] >= min_improvement
        )
        if is_good:
            good.append(result)
    pool = good if good else [r for r in results if r is not None]
    if not pool:
        return None, False, 0
    offsets = torch.stack([r["offset"] for r in pool])
    aggregate = {
        "offset": offsets.median(dim=0).values.float(),
        "bias": torch.stack([r["bias"] for r in pool]).median(dim=0).values.float(),
        "residual_zero": torch.stack([r["residual_zero"] for r in pool]).mean().float(),
        "residual_fit": torch.stack([r["residual_fit"] for r in pool]).mean().float(),
        "residual_improvement": torch.stack([r["residual_improvement"] for r in pool]).mean().float(),
        "condition_number": torch.stack([r["condition_number"] for r in pool]).median().float(),
        "observability_score": torch.stack([r["observability_score"] for r in pool]).median().float(),
        "window_consistency": offsets.std(dim=0).norm().float() if offsets.shape[0] > 1 else torch.tensor(0.0),
        "num_good_windows": int(len(good)),
        "num_total_windows": int(len([r for r in results if r is not None])),
    }
    outlier = not bool(good)
    return aggregate, outlier, len(good)


def aligned_sensor_inputs(seq, sensor_idx, dt_frames, use_ideal_vertex_acc=False):
    a_obs = seq["ddot_p_wv"][:, sensor_idx] if use_ideal_vertex_acc else seq["aM"][:, sensor_idx]
    ddot_p_wj = seq["ddot_p_wj"][:, sensor_idx]
    ddot_R_wj = seq["ddot_R_wj"][:, sensor_idx]
    if dt_frames > 0:
        return ddot_p_wj[:-dt_frames], ddot_R_wj[:-dt_frames], a_obs[dt_frames:]
    if dt_frames < 0:
        k = -dt_frames
        return ddot_p_wj[k:], ddot_R_wj[k:], a_obs[:-k]
    return ddot_p_wj, ddot_R_wj, a_obs


def window_record(start, end, result):
    if result is None:
        return {"start": int(start), "end": int(end), "valid": False}
    return {
        "start": int(start),
        "end": int(end),
        "valid": True,
        "offset": result["offset"].float(),
        "offset_norm": result["offset"].norm().float(),
        "bias": result["bias"].float(),
        "residual_zero": result["residual_zero"].float(),
        "residual_fit": result["residual_fit"].float(),
        "residual_improvement": result["residual_improvement"].float(),
        "condition_number": result["condition_number"].float(),
        "singular_values": result["singular_values"].float(),
        "observability_score": result["observability_score"].float(),
        "num_valid_frames": int(result["num_valid_frames"]),
    }


def empty_aggregate():
    return {
        "offset": torch.full((3,), float("nan")),
        "bias": torch.full((3,), float("nan")),
        "residual_zero": torch.tensor(float("nan")),
        "residual_fit": torch.tensor(float("nan")),
        "residual_improvement": torch.tensor(float("nan")),
        "condition_number": torch.tensor(float("inf")),
        "observability_score": torch.tensor(0.0),
        "window_consistency": torch.tensor(float("nan")),
        "num_good_windows": 0,
        "num_total_windows": 0,
    }


def estimate_sequence(seq, args, use_ideal_vertex_acc=False, dt_frames=0):
    n = seq["aM"].shape[0] - abs(int(dt_frames))
    ranges = window_ranges(n, args.window_size, args.stride) if n >= 6 else []
    sensor_outputs = []
    window_records = []
    for sensor_idx in range(6):
        ddot_p_wj, ddot_R_wj, a_obs = aligned_sensor_inputs(
            seq, sensor_idx, int(dt_frames), use_ideal_vertex_acc=use_ideal_vertex_acc
        )
        per_window = [
            solve_sensor_offset(
                ddot_p_wj,
                ddot_R_wj,
                a_obs,
                start,
                end,
                ridge=args.ridge,
                fit_bias=args.fit_bias,
                huber_delta=args.huber_delta,
                irls_iters=args.irls_iters,
            )
            for start, end in ranges
        ]
        aggregate, outlier, _ = aggregate_window_results(
            per_window,
            args.max_offset_norm,
            args.max_condition,
            args.min_improvement,
        )
        if aggregate is None:
            aggregate = empty_aggregate()
            outlier = True
        sensor_outputs.append((aggregate, outlier))
        window_records.append([window_record(start, end, result) for (start, end), result in zip(ranges, per_window)])
    return sensor_outputs, window_records


def quality_mask_from_metrics(
    offset_norm,
    residual_improvement,
    condition_number,
    observability_score,
    window_consistency,
    args,
):
    return (
        torch.isfinite(offset_norm)
        & torch.isfinite(residual_improvement)
        & torch.isfinite(condition_number)
        & torch.isfinite(observability_score)
        & torch.isfinite(window_consistency)
        & (offset_norm <= args.quality_max_offset_norm)
        & (residual_improvement >= args.quality_min_improvement)
        & (condition_number <= args.quality_max_condition)
        & (observability_score >= args.quality_min_observability)
        & (window_consistency <= args.quality_max_window_consistency)
    )


def dt_sensitivity_sequence(seq, args, use_ideal_vertex_acc=False):
    dt_values = [int(x) for x in args.dt_values.split(",") if x.strip()]
    per_dt = []
    offsets_by_dt = []
    for dt in dt_values:
        sensor_outputs, _ = estimate_sequence(seq, args, use_ideal_vertex_acc=use_ideal_vertex_acc, dt_frames=dt)
        offsets = torch.stack([item[0]["offset"] for item in sensor_outputs])
        offsets_by_dt.append(offsets)
        per_dt.append(
            {
                "dt": int(dt),
                "offset_norm": offsets.norm(dim=-1),
                "residual_fit": torch.stack([item[0]["residual_fit"] for item in sensor_outputs]),
                "residual_improvement": torch.stack([item[0]["residual_improvement"] for item in sensor_outputs]),
                "condition_number": torch.stack([item[0]["condition_number"] for item in sensor_outputs]),
                "observability_score": torch.stack([item[0]["observability_score"] for item in sensor_outputs]),
            }
        )
    residual_fit = torch.stack([item["residual_fit"] for item in per_dt])
    finite_fit = torch.where(torch.isfinite(residual_fit), residual_fit, torch.full_like(residual_fit, float("inf")))
    best_idx = torch.argmin(finite_fit, dim=0)
    dt_tensor = torch.tensor(dt_values, dtype=torch.long)
    offsets_by_dt = torch.stack(offsets_by_dt)
    zero_idx = dt_values.index(0) if 0 in dt_values else None
    offset_deviation_from_zero = (
        (offsets_by_dt - offsets_by_dt[zero_idx : zero_idx + 1]).norm(dim=-1)
        if zero_idx is not None
        else torch.full((len(dt_values), 6), float("nan"))
    )
    return {
        "dt_values": dt_tensor,
        "dt_best": dt_tensor[best_idx],
        "offset_norm": torch.stack([item["offset_norm"] for item in per_dt]),
        "residual_fit": residual_fit,
        "residual_improvement": torch.stack([item["residual_improvement"] for item in per_dt]),
        "condition_number": torch.stack([item["condition_number"] for item in per_dt]),
        "observability_score": torch.stack([item["observability_score"] for item in per_dt]),
        "offset_deviation_from_zero": offset_deviation_from_zero,
    }


def build_output(data, args, dataset_name, split, source_path, use_ideal_vertex_acc=False):
    max_sequences = len(data["pose"]) if args.max_sequences <= 0 else min(args.max_sequences, len(data["pose"]))
    names = []
    offsets = []
    biases = []
    residual_zero = []
    residual_fit = []
    residual_improvement = []
    condition_number = []
    observability_score = []
    offset_norm = []
    outlier_mask = []
    window_consistency = []
    synthetic_truth = []
    synthetic_error = []
    num_good_windows = []
    num_total_windows = []
    quality_masks = []
    window_records = []
    dt_summaries = []

    for seq_idx in range(max_sequences):
        seq = prepare_sequence_kinematics(
            data,
            seq_idx,
            smooth_window=args.smooth_window,
            max_frames=args.max_frames,
            device=args.device,
            derivative_mode=args.derivative_mode,
            smoothing_mode=args.smoothing_mode,
            acceleration_model=args.acceleration_model,
        )
        sensor_outputs, seq_window_records = estimate_sequence(seq, args, use_ideal_vertex_acc=use_ideal_vertex_acc)
        seq_offsets = torch.stack([item[0]["offset"] for item in sensor_outputs])
        seq_offset_norm = seq_offsets.norm(dim=-1)
        seq_residual_improvement = torch.stack([item[0]["residual_improvement"] for item in sensor_outputs])
        seq_condition_number = torch.stack([item[0]["condition_number"] for item in sensor_outputs])
        seq_observability_score = torch.stack([item[0]["observability_score"] for item in sensor_outputs])
        seq_window_consistency = torch.stack([item[0]["window_consistency"] for item in sensor_outputs])
        names.append(seq["name"])
        offsets.append(seq_offsets)
        biases.append(torch.stack([item[0]["bias"] for item in sensor_outputs]))
        residual_zero.append(torch.stack([item[0]["residual_zero"] for item in sensor_outputs]))
        residual_fit.append(torch.stack([item[0]["residual_fit"] for item in sensor_outputs]))
        residual_improvement.append(seq_residual_improvement)
        condition_number.append(seq_condition_number)
        observability_score.append(seq_observability_score)
        offset_norm.append(seq_offset_norm)
        outlier_mask.append(torch.tensor([item[1] for item in sensor_outputs], dtype=torch.bool))
        window_consistency.append(seq_window_consistency)
        num_good_windows.append(torch.tensor([item[0]["num_good_windows"] for item in sensor_outputs], dtype=torch.long))
        num_total_windows.append(torch.tensor([item[0]["num_total_windows"] for item in sensor_outputs], dtype=torch.long))
        quality_masks.append(
            quality_mask_from_metrics(
                seq_offset_norm,
                seq_residual_improvement,
                seq_condition_number,
                seq_observability_score,
                seq_window_consistency,
                args,
            )
        )
        window_records.append(seq_window_records)
        if args.dt_sensitivity:
            dt_summaries.append(dt_sensitivity_sequence(seq, args, use_ideal_vertex_acc=use_ideal_vertex_acc))
        if use_ideal_vertex_acc:
            truth_frames = theoretical_vertex_offsets(seq["p_wj"], seq["R_wj"], seq["p_wv"])
            truth = truth_frames.median(dim=0).values
            synthetic_truth.append(truth)
            synthetic_error.append((seq_offsets - truth).norm(dim=-1))
        print(f"[{seq_idx + 1}/{max_sequences}] {seq['name']} offset_norm={seq_offsets.norm(dim=-1).tolist()}", flush=True)

    output = {
        "name": names,
        "offset": torch.stack(offsets),
        "bias": torch.stack(biases),
        "residual_zero": torch.stack(residual_zero),
        "residual_fit": torch.stack(residual_fit),
        "residual_improvement": torch.stack(residual_improvement),
        "condition_number": torch.stack(condition_number),
        "observability_score": torch.stack(observability_score),
        "offset_norm": torch.stack(offset_norm),
        "outlier_mask": torch.stack(outlier_mask),
        "window_consistency": torch.stack(window_consistency),
        "num_good_windows": torch.stack(num_good_windows),
        "num_total_windows": torch.stack(num_total_windows),
        "quality_mask": torch.stack(quality_masks),
        "window_records": window_records,
        "metadata": make_metadata(dataset_name, split, source_path, args),
    }
    output["metadata"]["aggregation"] = {
        "sequence_offset": "coordinate-wise median over accepted windows; if no windows pass filters, median over valid windows and outlier_mask=True",
        "accepted_window_filters": {
            "max_offset_norm": args.max_offset_norm,
            "max_condition": args.max_condition,
            "min_improvement": args.min_improvement,
        },
        "quality_mask_filters": {
            "quality_max_offset_norm": args.quality_max_offset_norm,
            "quality_min_improvement": args.quality_min_improvement,
            "quality_max_condition": args.quality_max_condition,
            "quality_min_observability": args.quality_min_observability,
            "quality_max_window_consistency": args.quality_max_window_consistency,
        },
        "sensor_names": list(SENSOR_NAMES),
    }
    if args.dt_sensitivity:
        output["dt_sensitivity"] = dt_summaries
    if use_ideal_vertex_acc:
        output["synthetic_truth_offset"] = torch.stack(synthetic_truth)
        output["synthetic_offset_error"] = torch.stack(synthetic_error)
        output["metadata"]["synthetic_sanity"] = (
            "Uses ideal IMU vertex acceleration from saved v_imu/FK, not noisy synthetic aM. "
            "This checks the LS estimator against known joint-local vertex offsets."
        )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Offline diagnostic estimator for joint-local IMU position offsets r_JS.")
    parser.add_argument("--input", required=True, help="Path to GlobalPose-format .pt file.")
    parser.add_argument("--dataset", required=True, choices=("dip", "totalcapture", "amass", "synthetic"))
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smoothing-mode", default="moving_average", choices=("none", "moving_average", "centered_moving_average", "savgol"))
    parser.add_argument("--derivative-mode", default="legacy", choices=("legacy", "centered", "strict_centered"))
    parser.add_argument("--acceleration-model", default="ddot_R", choices=("ddot_R", "alpha_omega"))
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--fit-bias", action="store_true")
    parser.add_argument("--huber-delta", type=float, default=0.0)
    parser.add_argument("--irls-iters", type=int, default=5)
    parser.add_argument("--max-offset-norm", type=float, default=0.25)
    parser.add_argument("--max-condition", type=float, default=1e8)
    parser.add_argument("--min-improvement", type=float, default=-0.05)
    parser.add_argument("--quality-max-offset-norm", type=float, default=0.5)
    parser.add_argument("--quality-min-improvement", type=float, default=0.05)
    parser.add_argument("--quality-max-condition", type=float, default=1e8)
    parser.add_argument("--quality-min-observability", type=float, default=1e-6)
    parser.add_argument("--quality-max-window-consistency", type=float, default=0.15)
    parser.add_argument("--dt-sensitivity", action="store_true")
    parser.add_argument("--dt-values", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--synthetic-sanity", action="store_true", help="Use ideal vertex acceleration and report known r_JS recovery error.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset_file(args.input)
    output = build_output(
        data,
        args,
        dataset_name=args.dataset,
        split=args.split,
        source_path=args.input,
        use_ideal_vertex_acc=args.synthetic_sanity,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"Saved offset diagnostics to {out_path}")
    print(f"offset shape: {tuple(output['offset'].shape)} fps={FPS}")
    if args.synthetic_sanity:
        err = output["synthetic_offset_error"]
        print(f"synthetic offset error mean={err.mean().item():.6f} m max={err.max().item():.6f} m")


if __name__ == "__main__":
    main()
