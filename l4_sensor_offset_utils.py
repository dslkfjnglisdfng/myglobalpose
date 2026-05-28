import json
import inspect
from pathlib import Path

import numpy as np
import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

import articulate as art


FPS = 60.0
DT = 1.0 / FPS
GRAVITY_WORLD = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float32)
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)
SENSOR_NAMES = (
    "left_forearm",
    "right_forearm",
    "left_lower_leg",
    "right_lower_leg",
    "head",
    "pelvis",
)
COORDINATE_CONTRACT = (
    "r_JS is the IMU origin position relative to the mapped joint J, expressed in joint-local "
    "coordinates. World prediction: p_WS(t)=p_WJ(t)+R_WJ(t)@r_JS."
)


_BODY_MODEL = None


def body_model(device="cpu"):
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _BODY_MODEL = art.ParametricModel("models/SMPL_male.pkl", vert_mask=IMU_VERTICES, device=torch.device(device))
    return _BODY_MODEL


def sensor_to_joint_map():
    return {
        "sensor_names": list(SENSOR_NAMES),
        "imu_joints": list(IMU_JOINTS),
        "imu_vertices": list(IMU_VERTICES),
        "coordinate_contract": COORDINATE_CONTRACT,
    }


def load_dataset_file(path):
    data = torch.load(path, map_location="cpu")
    required = ("pose", "tran")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    return data


def official_imu_fields(data, seq_idx):
    if all(key in data for key in ("aM", "wM", "RMB")):
        return (
            data["aM"][seq_idx].float(),
            data["wM"][seq_idx].float(),
            data["RMB"][seq_idx].float(),
        )
    for key in ("RIM", "RSB", "RIS", "aS", "wS"):
        if key not in data:
            raise KeyError(f"Dataset has neither aM/wM/RMB nor raw field {key}")
    gravity = GRAVITY_WORLD.view(1, 1, 3)
    RIM = data["RIM"][seq_idx].float()
    RSB = data["RSB"][seq_idx].float()
    RIS = data["RIS"][seq_idx].float()
    aS = data["aS"][seq_idx].float()
    wS = data["wS"][seq_idx].float()
    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + gravity
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1)
    return aM.float(), wM.float(), RMB.float()


def pose_to_rotation_matrices(pose):
    pose = pose.float()
    if pose.dim() == 2 and pose.shape[-1] == 72:
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3).float()
    if pose.dim() == 3 and pose.shape[-2:] == (24, 3):
        return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3).float()
    if pose.dim() == 4 and pose.shape[-3:] == (24, 3, 3):
        return pose.float()
    raise ValueError(f"Unsupported pose shape: {tuple(pose.shape)}")


@torch.no_grad()
def fk_imu_joints_and_vertices(pose, tran, device="cpu"):
    pose_R = pose_to_rotation_matrices(pose).to(device)
    tran = tran.float().view(-1, 3).to(device)
    model = body_model(device)
    grot, joint, vert = model.forward_kinematics(pose_R, None, tran, calc_mesh=True)
    imu_joint_pos = joint[:, IMU_JOINTS].detach().cpu().float()
    imu_joint_rot = grot[:, IMU_JOINTS].detach().cpu().float()
    imu_vertex_pos = vert.detach().cpu().float()
    return imu_joint_pos, imu_joint_rot, imu_vertex_pos


def moving_average(x, window):
    window = int(window)
    if window <= 1:
        return x.float()
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeeze = True
    pad = window // 2
    flat = x.reshape(x.shape[0], -1).transpose(0, 1).unsqueeze(0)
    flat = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    kernel = torch.ones(flat.shape[1], 1, window, dtype=flat.dtype, device=flat.device) / window
    out = torch.nn.functional.conv1d(flat, kernel, groups=flat.shape[1]).squeeze(0).transpose(0, 1)
    out = out.reshape_as(x)
    return out[:, 0] if squeeze else out


def savgol_smooth(x, window, polyorder=2):
    window = int(window)
    if window <= 1:
        return x.float()
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd")
    if window <= polyorder:
        raise ValueError("Savitzky-Golay window must be greater than polyorder")
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError("scipy.signal.savgol_filter is required for savgol smoothing") from exc
    original_shape = x.shape
    y = x.detach().cpu().float().reshape(x.shape[0], -1).numpy()
    y = savgol_filter(y, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    return torch.from_numpy(y.copy()).reshape(original_shape).to(dtype=x.dtype)


def smooth_centered(x, window, mode="moving_average"):
    if mode in ("none", "identity") or int(window) <= 1:
        return x.float()
    if mode in ("moving_average", "centered_moving_average"):
        return moving_average(x, window)
    if mode in ("savgol", "savitzky_golay"):
        return savgol_smooth(x, window)
    raise ValueError(f"Unsupported smoothing mode: {mode}")


def finite_difference_first(x, fps=FPS):
    x = x.float()
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    v = torch.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) * (0.5 * fps)
    v[0] = (x[1] - x[0]) * fps
    v[-1] = (x[-1] - x[-2]) * fps
    return v


def finite_difference_first_centered(x, fps=FPS):
    x = x.float()
    v = torch.full_like(x, float("nan"))
    if x.shape[0] < 3:
        return v
    v[1:-1] = (x[2:] - x[:-2]) * (0.5 * fps)
    return v


def finite_difference_second(x, fps=FPS):
    x = x.float()
    if x.shape[0] < 4:
        return torch.zeros_like(x)
    a = torch.zeros_like(x)
    a[1:-1] = (x[:-2] - 2.0 * x[1:-1] + x[2:]) * (fps ** 2)
    a[0] = (2.0 * x[0] - 5.0 * x[1] + 4.0 * x[2] - x[3]) * (fps ** 2)
    a[-1] = (2.0 * x[-1] - 5.0 * x[-2] + 4.0 * x[-3] - x[-4]) * (fps ** 2)
    return a


def finite_difference_second_centered(x, fps=FPS):
    x = x.float()
    a = torch.full_like(x, float("nan"))
    if x.shape[0] < 3:
        return a
    a[1:-1] = (x[:-2] - 2.0 * x[1:-1] + x[2:]) * (fps ** 2)
    return a


def second_derivative(x, fps=FPS, mode="legacy"):
    if mode in ("legacy", "old"):
        return finite_difference_second(x, fps=fps)
    if mode in ("centered", "strict_centered"):
        return finite_difference_second_centered(x, fps=fps)
    raise ValueError(f"Unsupported derivative mode: {mode}")


def skew_matrix(v):
    v = v.float()
    out = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def vee_skew(mat):
    mat = mat.float()
    return torch.stack((mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]), dim=-1)


def first_derivative(x, fps=FPS, mode="legacy"):
    if mode in ("legacy", "old"):
        return finite_difference_first(x, fps=fps)
    if mode in ("centered", "strict_centered"):
        return finite_difference_first_centered(x, fps=fps)
    raise ValueError(f"Unsupported derivative mode: {mode}")


def lever_arm_matrix_alpha_omega(R_wj, fps=FPS, derivative_mode="legacy"):
    """World-frame matrix A(t) such that lever acceleration is A(t) @ r_JS."""
    R_wj = R_wj.float()
    R_dot = first_derivative(R_wj, fps=fps, mode=derivative_mode)
    omega_hat = R_dot.matmul(R_wj.transpose(-1, -2))
    omega_hat = 0.5 * (omega_hat - omega_hat.transpose(-1, -2))
    omega = vee_skew(omega_hat)
    alpha = first_derivative(omega, fps=fps, mode=derivative_mode)
    omega_x = skew_matrix(omega)
    alpha_x = skew_matrix(alpha)
    return alpha_x.matmul(R_wj) + omega_x.matmul(omega_x).matmul(R_wj)


def prepare_sequence_kinematics(
    data,
    seq_idx,
    smooth_window=1,
    max_frames=0,
    device="cpu",
    derivative_mode="legacy",
    smoothing_mode="moving_average",
    acceleration_model="ddot_R",
):
    pose = data["pose"][seq_idx].float()
    tran = data["tran"][seq_idx].float()
    aM, wM, RMB = official_imu_fields(data, seq_idx)
    n = min(pose.shape[0], tran.shape[0], aM.shape[0], wM.shape[0], RMB.shape[0])
    if max_frames:
        n = min(n, int(max_frames))
    pose, tran = pose[:n], tran[:n]
    aM, wM, RMB = aM[:n], wM[:n], RMB[:n]
    p_wj, R_wj, p_wv = fk_imu_joints_and_vertices(pose, tran, device=device)
    if "v_imu" in data and isinstance(data["v_imu"], list) and seq_idx < len(data["v_imu"]):
        saved_v_imu = data["v_imu"][seq_idx].float()[:n]
        if saved_v_imu.shape[-2:] == (6, 3):
            p_wv = saved_v_imu
    if smooth_window > 1 and smoothing_mode not in ("none", "identity"):
        p_wj = smooth_centered(p_wj, smooth_window, mode=smoothing_mode)
        R_wj = smooth_centered(R_wj, smooth_window, mode=smoothing_mode)
        p_wv = smooth_centered(p_wv, smooth_window, mode=smoothing_mode)
        aM = smooth_centered(aM, smooth_window, mode=smoothing_mode)
    if acceleration_model in ("ddot_R", "matrix_second_derivative"):
        ddot_R_wj = second_derivative(R_wj, mode=derivative_mode)
    elif acceleration_model in ("alpha_omega", "rigid_body"):
        ddot_R_wj = lever_arm_matrix_alpha_omega(R_wj, derivative_mode=derivative_mode)
    else:
        raise ValueError(f"Unsupported acceleration model: {acceleration_model}")
    return {
        "name": str(data["name"][seq_idx]) if "name" in data else f"seq_{seq_idx}",
        "pose": pose,
        "tran": tran,
        "aM": aM,
        "wM": wM,
        "RMB": RMB,
        "p_wj": p_wj,
        "R_wj": R_wj,
        "p_wv": p_wv,
        "ddot_p_wj": second_derivative(p_wj, mode=derivative_mode),
        "ddot_R_wj": ddot_R_wj,
        "ddot_p_wv": second_derivative(p_wv, mode=derivative_mode),
        "fps": FPS,
        "derivative_mode": derivative_mode,
        "smoothing_mode": smoothing_mode,
        "smooth_window": int(smooth_window),
        "acceleration_model": acceleration_model,
    }


def theoretical_vertex_offsets(p_wj, R_wj, p_wv):
    return R_wj.transpose(-1, -2).matmul((p_wv - p_wj).unsqueeze(-1)).squeeze(-1)


def make_metadata(dataset_name, split, source_path, args=None):
    meta = {
        "dataset": dataset_name,
        "split": split,
        "source_path": str(source_path),
        "fps": FPS,
        "gravity_world": GRAVITY_WORLD.tolist(),
        "sensor_to_joint": sensor_to_joint_map(),
        "coordinate_contract": COORDINATE_CONTRACT,
        "units": {
            "position": "meters",
            "acceleration": "meters/second^2",
            "angular_velocity": "radians/second",
        },
    }
    if args is not None:
        meta["args"] = vars(args)
    return meta


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
