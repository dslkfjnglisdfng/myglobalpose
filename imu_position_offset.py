import json
from pathlib import Path

import torch
from torch.nn.functional import relu

from l4_sensor_offset_utils import (
    FPS,
    GRAVITY_WORLD,
    IMU_JOINTS,
    SENSOR_NAMES,
    body_model as smpl_body_model,
    fk_imu_joints_and_vertices,
    official_imu_fields,
    pose_to_rotation_matrices,
    second_derivative,
    smooth_centered,
)
from l4_rawlike_se3_calibration import matvec, robust_rotation_mean


OFFSET_POSITION_CONTRACT = (
    "r_JS is the IMU origin position relative to mapped joint J, expressed in "
    "joint-local coordinates. World prediction: p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS. "
    "For DIP-IMU, tran is not used as trusted global translation; acceleration "
    "objectives are diagnostic/self-supervised only."
)


FOOTLOCK_POSITION_CONTRACT = (
    OFFSET_POSITION_CONTRACT
    + " footlock_transpose_v1 estimates DIP pseudo-r_JS from short stance windows: "
    "a stance foot contact point C is assumed static in W, root translation is "
    "only used through p_WR(t)=-p_WC_zero_tran(t)+constant. Contact selection "
    "uses raw official aM/RMB; the lever-arm fit may smooth FK and aM. These "
    "pseudo offsets are not ground truth."
)


LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11
LEFT_LOWER_LEG_SENSOR = 2
RIGHT_LOWER_LEG_SENSOR = 3


class _TransposeRNN(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(self.dropout(x))).unsqueeze(1), h)
        return self.linear2(x.squeeze(1)), h


class TransPoseContactEstimator(torch.nn.Module):
    """Minimal TransPose pose_s1 + tran_b1 adapter for offline contact logits."""

    def __init__(self, weights_path, device="cpu"):
        super().__init__()
        self.pose_s1 = _TransposeRNN(72, 15, 256)
        self.tran_b1 = _TransposeRNN(87, 2, 64)
        weights = torch.load(weights_path, map_location=device)
        if isinstance(weights, dict) and "model_state_dict" in weights:
            weights = weights["model_state_dict"]
        self.pose_s1.load_state_dict(self._strip_prefix(weights, "pose_s1."))
        self.tran_b1.load_state_dict(self._strip_prefix(weights, "tran_b1."))
        self.to(device)
        self.eval()

    @staticmethod
    def _strip_prefix(weights, prefix):
        out = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        if not out:
            raise KeyError(f"TransPose weights missing prefix {prefix}")
        return out

    @torch.no_grad()
    def contact_probability(self, aM, RMB, acc_scale=30.0, device="cpu"):
        imu = torch.cat((aM.reshape(aM.shape[0], -1) / float(acc_scale), RMB.reshape(RMB.shape[0], -1)), dim=-1)
        imu = imu.to(device=device, dtype=torch.float32)
        leaf = self.pose_s1(imu)[0]
        logits = self.tran_b1(torch.cat((leaf, imu), dim=-1))[0]
        return torch.sigmoid(logits).detach().cpu()


def stack_if_list(value):
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack([item.float() for item in value])
    raise TypeError(type(value))


def load_offset_cache(path):
    cache = torch.load(path, map_location="cpu")
    names = cache.get("name") or cache.get("sequence_id")
    if names is None:
        raise KeyError(f"{path} missing name/sequence_id")
    if "offset" in cache:
        offsets = cache["offset"]
    elif "r_JS" in cache:
        offsets = cache["r_JS"]
    elif "imu_offset_r" in cache:
        offsets = cache["imu_offset_r"]
    else:
        raise KeyError(f"{path} missing offset/r_JS/imu_offset_r")
    offsets = stack_if_list(offsets)
    if offsets.shape[1:] != (6, 3):
        raise ValueError(f"{path} offset shape={tuple(offsets.shape)}, expected [N,6,3]")
    return {str(name): offsets[idx].float() for idx, name in enumerate(names)}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def rotation_angle_deg(R):
    trace = R.float().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def prepare_sequence(data, seq_idx, device="cpu", smooth_window=5, derivative_mode="centered", max_frames=0):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    aM, wM, RMB = official_imu_fields(data, seq_idx)
    n = min(pose.shape[0], tran.shape[0], aM.shape[0], wM.shape[0], RMB.shape[0])
    if int(max_frames) > 0:
        n = min(n, int(max_frames))
    pose, tran = pose[:n], tran[:n]
    aM, wM, RMB = aM[:n], wM[:n], RMB[:n]
    contact_aM = aM.float().clone()
    contact_RMB = RMB.float().clone()
    p_wj, R_wj, _ = fk_imu_joints_and_vertices(pose, tran, device=device)
    if smooth_window > 1:
        p_wj = smooth_centered(p_wj, smooth_window)
        R_wj = smooth_centered(R_wj, smooth_window)
        aM = smooth_centered(aM, smooth_window)
    R_js_frames = R_wj.transpose(-1, -2).matmul(RMB.float())
    R_js = torch.stack([robust_rotation_mean(R_js_frames[:, sensor_idx]) for sensor_idx in range(6)])
    return {
        "name": str(data["name"][seq_idx]) if "name" in data else f"seq_{seq_idx}",
        "pose": pose,
        "tran": tran,
        "aM": aM.float(),
        "wM": wM.float(),
        "RMB": RMB.float(),
        "contact_aM": contact_aM,
        "contact_RMB": contact_RMB,
        "fit_smooth_window": int(smooth_window),
        "fit_derivative_mode": derivative_mode,
        "p_wj": p_wj.float(),
        "R_wj": R_wj.float(),
        "R_JS": R_js.float(),
        "ddot_p_wj": second_derivative(p_wj, fps=FPS, mode=derivative_mode).float(),
        "ddot_R_wj": second_derivative(R_wj, fps=FPS, mode=derivative_mode).float(),
    }


@torch.no_grad()
def _fk_zero_translation_full_joints(pose, device="cpu"):
    pose_R = pose_to_rotation_matrices(pose).to(device)
    tran = torch.zeros(pose_R.shape[0], 3, dtype=pose_R.dtype, device=pose_R.device)
    model = smpl_body_model(device)
    grot, joint = model.forward_kinematics(pose_R, None, tran, calc_mesh=False)
    return joint.detach().cpu().float(), grot.detach().cpu().float()


def _split_contact_mask(mask, min_frames, max_frames):
    mask = mask.bool().cpu()
    windows = []
    start = None
    for idx, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx
            while end - start >= int(min_frames):
                chunk_end = min(end, start + int(max_frames))
                if chunk_end - start >= int(min_frames):
                    windows.append((start, chunk_end))
                start = chunk_end
            start = None
    return windows


def _transpose_contact_weight(prob, low=0.5, high=0.9):
    return ((prob - float(low)) / (float(high) - float(low))).clamp(0.0, 1.0)


def select_footlock_windows(
    contact_prob,
    threshold=0.85,
    margin=0.15,
    min_frames=24,
    max_frames=180,
    foot_pos=None,
    selection_mode="transpose_winner",
    contact_height_margin=0.08,
    transpose_prob_low=0.5,
    transpose_prob_high=0.9,
):
    if selection_mode != "transpose_winner":
        raise ValueError("Only contact_selection_mode=transpose_winner is active for footlock_transpose_v1.")
    windows = []
    max_prob, winner = contact_prob.max(dim=1)
    contact_weight = _transpose_contact_weight(max_prob, low=transpose_prob_low, high=transpose_prob_high)
    base = max_prob >= float(threshold)
    if foot_pos is not None and float(contact_height_margin) >= 0:
        foot_y = foot_pos[:, :, 1]
        near_lowest = foot_y <= foot_y.min(dim=1, keepdim=True).values + float(contact_height_margin)
    else:
        near_lowest = torch.ones_like(contact_prob, dtype=torch.bool)
    side_specs = (
        ("left", LEFT_FOOT_JOINT, (winner == 0) & base & near_lowest[:, 0]),
        ("right", RIGHT_FOOT_JOINT, (winner == 1) & base & near_lowest[:, 1]),
    )
    for side, foot_joint, mask in side_specs:
        side_windows = _split_contact_mask(mask, min_frames, max_frames)
        selection = "transpose_winner_height"
        if not side_windows:
            fallback_mask = (winner == (0 if side == "left" else 1)) & base
            side_windows = _split_contact_mask(fallback_mask, min_frames, max_frames)
            selection = "transpose_winner_probability"
        for start, end in side_windows:
            windows.append({
                "side": side,
                "start": int(start),
                "end": int(end),
                "foot_joint": foot_joint,
                "selection": selection,
                "mean_contact_weight": float(contact_weight[start:end].mean()) if end > start else 0.0,
            })
    windows.sort(key=lambda item: (item["start"], item["side"]))
    return windows


def _solve_lever_rows(A, y, ridge=1e-3, fit_sensor_bias=True):
    if fit_sensor_bias:
        eye = torch.eye(3, dtype=A.dtype).view(1, 3, 3).expand(A.shape[0], -1, -1)
        M = torch.cat((A, eye), dim=-1).reshape(-1, 6)
        reg = torch.diag(torch.tensor([ridge, ridge, ridge, 1000.0, 1000.0, 1000.0], dtype=A.dtype))
    else:
        M = A.reshape(-1, 3)
        reg = torch.eye(3, dtype=A.dtype) * ridge
    target = y.reshape(-1)
    lhs = M.T.matmul(M) + reg
    rhs = M.T.matmul(target)
    try:
        sol = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        sol = torch.linalg.lstsq(lhs, rhs).solution
    r = sol[:3] if fit_sensor_bias else sol
    b = sol[3:] if fit_sensor_bias else torch.zeros(3, dtype=A.dtype)
    pred = A.matmul(r.view(3, 1)).squeeze(-1) + b.view(1, 3)
    svals = torch.linalg.svdvals(A.reshape(-1, 3))
    cond = svals.max() / svals.min().clamp_min(1e-12)
    return {
        "offset": r.float(),
        "acc_bias": b.float(),
        "residual_zero": y.norm(dim=-1).mean().float(),
        "residual_fit": (y - pred).norm(dim=-1).mean().float(),
        "condition_number": cond.float(),
    }


def _sensor_skip_for_window(window):
    if window["side"] == "left":
        return {LEFT_LOWER_LEG_SENSOR}
    if window["side"] == "right":
        return {RIGHT_LOWER_LEG_SENSOR}
    return set()


def solve_footlock_transpose_offset(
    seq,
    contact_estimator,
    device="cpu",
    ridge=1e-3,
    fit_sensor_bias=True,
    max_offset_norm=0.5,
    contact_threshold=0.85,
    contact_margin=0.15,
    min_contact_frames=24,
    max_contact_frames=180,
    contact_selection_mode="transpose_winner",
    contact_height_margin=0.08,
    transpose_prob_low=0.5,
    transpose_prob_high=0.9,
    min_fit_frames=48,
    min_fit_improvement=0.05,
    max_condition_number=1e5,
    fallback_offset=None,
    derivative_mode="centered",
    smooth_window=5,
):
    contact_prob = contact_estimator.contact_probability(
        seq.get("contact_aM", seq["aM"]),
        seq.get("contact_RMB", seq["RMB"]),
        device=device,
    )
    joint_zero, grot_zero = _fk_zero_translation_full_joints(seq["pose"], device=device)
    p_wj_zero = joint_zero[:, list(IMU_JOINTS)]
    R_wj = grot_zero[:, list(IMU_JOINTS)]
    foot_pos = joint_zero[:, [LEFT_FOOT_JOINT, RIGHT_FOOT_JOINT]]
    if smooth_window > 1:
        p_wj_zero = smooth_centered(p_wj_zero, smooth_window)
        R_wj = smooth_centered(R_wj, smooth_window)
        foot_pos = smooth_centered(foot_pos, smooth_window)

    windows = select_footlock_windows(
        contact_prob,
        threshold=contact_threshold,
        margin=contact_margin,
        min_frames=min_contact_frames,
        max_frames=max_contact_frames,
        foot_pos=foot_pos,
        selection_mode=contact_selection_mode,
        contact_height_margin=contact_height_margin,
        transpose_prob_low=transpose_prob_low,
        transpose_prob_high=transpose_prob_high,
    )

    R_js_frames = R_wj.transpose(-1, -2).matmul(seq["RMB"].float())
    R_js = torch.stack([robust_rotation_mean(R_js_frames[:, sensor_idx]) for sensor_idx in range(6)])
    fallback = torch.zeros(6, 3) if fallback_offset is None else fallback_offset.float().view(6, 3)

    A_parts = [[] for _ in range(6)]
    y_parts = [[] for _ in range(6)]
    frame_counts = [0 for _ in range(6)]
    window_counts = [0 for _ in range(6)]
    side_counts = {"left": 0, "right": 0}

    for window in windows:
        start, end = window["start"], window["end"]
        side_counts[window["side"]] += 1
        foot_col = 0 if window["side"] == "left" else 1
        root_tran = -foot_pos[start:end, foot_col]
        p_wj = p_wj_zero[start:end] + root_tran[:, None, :]
        ddot_p_wj = second_derivative(p_wj, fps=FPS, mode=derivative_mode)
        ddot_R_wj = second_derivative(R_wj[start:end], fps=FPS, mode=derivative_mode)
        skip = _sensor_skip_for_window(window)
        for sensor_idx in range(6):
            if sensor_idx in skip:
                continue
            R_ws_t = R_wj[start:end, sensor_idx].matmul(R_js[sensor_idx]).transpose(-1, -2)
            c = matvec(R_ws_t, ddot_p_wj[:, sensor_idx] - GRAVITY_WORLD.view(1, 3))
            A = R_ws_t.matmul(ddot_R_wj[:, sensor_idx])
            y = seq["aM"][start:end, sensor_idx] - c
            valid = torch.isfinite(A).all(dim=(-1, -2)) & torch.isfinite(y).all(dim=-1)
            if valid.any():
                A_parts[sensor_idx].append(A[valid])
                y_parts[sensor_idx].append(y[valid])
                frame_counts[sensor_idx] += int(valid.sum())
                window_counts[sensor_idx] += 1

    offsets = []
    biases = []
    residual_zero = []
    residual_fit = []
    conditions = []
    confidence = []
    fallback_reason = []
    fit_improvement = []
    for sensor_idx in range(6):
        reason = ""
        if frame_counts[sensor_idx] < int(min_fit_frames) or not A_parts[sensor_idx]:
            reason = "insufficient_contact_equations"
        if reason:
            offsets.append(fallback[sensor_idx])
            biases.append(torch.zeros(3))
            residual_zero.append(torch.tensor(float("nan")))
            residual_fit.append(torch.tensor(float("nan")))
            conditions.append(torch.tensor(float("inf")))
            confidence.append(torch.tensor(0.0))
            fallback_reason.append(reason)
            fit_improvement.append(torch.tensor(float("nan")))
            continue

        A = torch.cat(A_parts[sensor_idx], dim=0)
        y = torch.cat(y_parts[sensor_idx], dim=0)
        solved = _solve_lever_rows(A, y, ridge=ridge, fit_sensor_bias=fit_sensor_bias)
        res_zero = solved["residual_zero"]
        res_fit = solved["residual_fit"]
        improvement = (res_zero - res_fit) / res_zero.clamp_min(1e-12)
        raw_norm = solved["offset"].norm()
        if not torch.isfinite(solved["condition_number"]) or solved["condition_number"] > float(max_condition_number):
            reason = "ill_conditioned"
        elif not torch.isfinite(improvement) or improvement < float(min_fit_improvement):
            reason = "low_residual_improvement"
        elif not torch.isfinite(raw_norm) or raw_norm > float(max_offset_norm):
            reason = "offset_norm_out_of_range"

        if reason:
            offsets.append(fallback[sensor_idx])
            biases.append(torch.zeros(3))
            confidence.append(torch.tensor(0.0))
        else:
            offsets.append(solved["offset"])
            biases.append(solved["acc_bias"])
            cond_score = (float(max_condition_number) / solved["condition_number"].clamp_min(1.0)).clamp(max=1.0)
            frame_score = torch.tensor(frame_counts[sensor_idx] / (frame_counts[sensor_idx] + float(min_fit_frames)))
            confidence.append((improvement.clamp(0.0, 1.0) * cond_score * frame_score).float())
        residual_zero.append(res_zero)
        residual_fit.append(res_fit)
        conditions.append(solved["condition_number"])
        fallback_reason.append(reason)
        fit_improvement.append(improvement.float())

    return {
        "offset": torch.stack(offsets).float(),
        "acc_bias": torch.stack(biases).float(),
        "residual_zero": torch.stack(residual_zero).float(),
        "residual_fit": torch.stack(residual_fit).float(),
        "condition_number": torch.stack(conditions).float(),
        "confidence": torch.stack(confidence).float(),
        "fit_improvement": torch.stack(fit_improvement).float(),
        "num_fit_frames": torch.tensor(frame_counts, dtype=torch.float32),
        "num_fit_windows": torch.tensor(window_counts, dtype=torch.float32),
        "fallback_reason": fallback_reason,
        "contact_probability_mean": contact_prob.mean(dim=0).float(),
        "contact_selection_mode": contact_selection_mode,
        "contact_height_margin": float(contact_height_margin),
        "contact_window_count": len(windows),
        "contact_side_window_count": side_counts,
        "contact_windows": windows,
        "contact_input": "raw_official_aM_RMB",
        "fit_input": f"smoothed_aM_and_zero_translation_FK_window_{smooth_window}",
        "source": "footlock_transpose_contact_pseudo_rjs",
        "coordinate_contract": FOOTLOCK_POSITION_CONTRACT,
    }


def plausibility_project(offset, max_norm=0.5):
    offset = offset.float().view(6, 3)
    norm = offset.norm(dim=-1, keepdim=True)
    scale = torch.clamp(torch.as_tensor(max_norm, dtype=offset.dtype) / norm.clamp_min(1e-12), max=1.0)
    return offset * scale


def build_output(names, offsets, rows, method, source_path, extra=None):
    offset_tensor = torch.stack(offsets).float()
    output = {
        "name": names,
        "offset": offset_tensor,
        "r_JS": offset_tensor,
        "imu_offset_r": offset_tensor,
        "method": method,
        "coordinate_contract": OFFSET_POSITION_CONTRACT,
        "sensor_names": list(SENSOR_NAMES),
        "imu_joints": list(IMU_JOINTS),
        "source_path": str(source_path),
        "rows": rows,
    }
    if extra:
        output.update(extra)
    return output
