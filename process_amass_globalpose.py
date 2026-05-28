"""
Process AMASS into GlobalPose-style synthetic IMU training data.

This script keeps the official GlobalPose baseline files untouched and
combines the two official data-processing pieces:

1. `process.py:process_amass()` for AMASS subset selection, 60 FPS
   resampling, and AMASS-to-DIP/SMPL frame alignment.
2. `imu_synthesis.py:syn_imu_from_smpl()` / `_syn_imu()` for GlobalPose's
   noisy synthetic IMU generation with calibration-error modeling.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import articulate as art


AMASS_DATASETS = [
    "HumanEva",
    "MPI_HDM05",
    "SFU",
    "MPI_mosh",
    "Transitions_mocap",
    "SSM_synced",
    "CMU",
    "DFaust67",
    "Eyes_Japan_Dataset",
    "KIT",
    "BMLmovi",
    "EKUT",
    "TCD_handMocap",
    "ACCAD",
    "BioMotionLab_NTroje",
    "BMLhandball",
    "MPI_Limits",
    "TotalCapture",
]

AMASS_DATASET_ALIASES = {
    "DFaust67": ("DFaust67", "DFaust_67"),
}

AMASS_TO_SMPL_FRAME = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]])
GRAVITY_WORLD = (0, -9.8, 0)
MAGNETIC_WORLD = (1.0, 0.0, 0.0)
GRAVITY = torch.tensor(GRAVITY_WORLD, dtype=torch.float32)
MAGNETIC_FIELD = torch.tensor(MAGNETIC_WORLD, dtype=torch.float32)
IMU_VERTICES = (1961, 5424, 1176, 4662, 411, 3021)
IMU_JOINTS = (18, 19, 4, 5, 15, 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model = art.ParametricModel("models/SMPL_male.pkl", vert_mask=IMU_VERTICES, device=device)


def _walking_noise(shape, std):
    return torch.cumsum(torch.normal(torch.zeros(shape), std), dim=0)


def _forward_smpl(pose, tran):
    pose = art.math.axis_angle_to_rotation_matrix(pose.view(-1, 24, 3)).view(-1, 24, 3, 3).to(device)
    tran = tran.view(-1, 3).to(device)
    grot = torch.empty(0, 24, 3, 3, device=device)
    joint = torch.empty(0, 24, 3, device=device)
    vert = torch.empty(0, 6, 3, device=device)
    for p_batch, t_batch in zip(pose.split(800), tran.split(800)):
        grot_, joint_, vert_ = body_model.forward_kinematics(p_batch, None, t_batch, calc_mesh=True)
        grot = torch.cat((grot, grot_), dim=0)
        joint = torch.cat((joint, joint_), dim=0)
        vert = torch.cat((vert, vert_), dim=0)
    return grot, joint, vert


def _syn_imu(p_imu_mount, R_imu_body, use_eskf=False):
    """GlobalPose official synthetic IMU logic, with ESKF loaded only on demand."""
    from articulate.utils.imu import IMUSimulator

    num_frames = len(p_imu_mount)
    k = np.sqrt(np.pi / 8)
    RBS = art.math.generate_random_rotation_matrix(6).to(device)
    dp = _walking_noise(shape=(num_frames, 6, 3), std=1e-3 * k * np.sqrt(1 / 60)) + torch.randn(6, 3) * 1e-2 * k
    dw = _walking_noise(shape=(num_frames, 6, 3), std=1e-2 * k * np.sqrt(1 / 60)) + torch.randn(6, 3) * 1e-1 * k
    dR = art.math.axis_angle_to_rotation_matrix(dw).view(-1, 6, 3, 3)
    p_imu = p_imu_mount + R_imu_body.matmul(dp.unsqueeze(-1).to(device)).squeeze(-1)
    R_imu = R_imu_body.matmul(RBS).matmul(dR.to(device))

    imu_simulator = IMUSimulator()
    imu_simulator.set_trajectory(p_imu, R_imu, fps=60)
    aS = imu_simulator.get_acceleration(gW=GRAVITY_WORLD)
    wS = imu_simulator.get_angular_velocity()
    mS = imu_simulator.get_magnetic_field(mW=MAGNETIC_WORLD)

    aS = torch.normal(aS, std=5e-2) + _walking_noise(shape=(num_frames, 6, 3), std=1e-4 * np.sqrt(1 / 60)).view_as(aS).to(device)
    wS = torch.normal(wS, std=5e-3) + _walking_noise(shape=(num_frames, 6, 3), std=1e-5 * np.sqrt(1 / 60)).view_as(wS).to(device)
    mS = torch.normal(mS, std=5e-3) + _walking_noise(shape=(num_frames, 6, 3), std=1e-5 * np.sqrt(1 / 60)).view_as(mS).to(device)

    R_sim = torch.empty(num_frames, 6, 3, 3)
    if use_eskf:
        import carticulate as cart

        for imu_idx in range(6):
            eskf = cart.ESKF(an=5e-2, wn=5e-3, aw=1e-4, ww=1e-5, mn=5e-3)
            eskf.initialize_9dof(RIS=R_imu[0, imu_idx].cpu().numpy(), gI=np.array(GRAVITY_WORLD), nI=np.array(MAGNETIC_WORLD))
            for frame_idx in range(num_frames):
                eskf.predict(am=aS[frame_idx, imu_idx].cpu().numpy(), wm=wS[frame_idx, imu_idx].cpu().numpy(), dt=1 / 60)
                eskf.correct(am=aS[frame_idx, imu_idx].cpu().numpy(), wm=wS[frame_idx, imu_idx].cpu().numpy(), mm=mS[frame_idx, imu_idx].cpu().numpy())
                R_sim[frame_idx, imu_idx] = torch.from_numpy(eskf.get_orientation_R())
    else:
        dR = art.math.axis_angle_to_rotation_matrix(wS / 60).view(-1, 6, 3, 3).cpu()
        R_sim[0] = R_imu[0].cpu()
        for frame_idx in range(1, num_frames):
            R_sim[frame_idx] = R_sim[frame_idx - 1].matmul(dR[frame_idx])

    nR = art.math.axis_angle_to_rotation_matrix(torch.randn(num_frames, 6, 3) * 0.1 * k).view(-1, 6, 3, 3)
    R_sim = R_sim.matmul(nR)

    R_sim = R_sim.to(device)
    a_sim = R_sim.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor(GRAVITY_WORLD, device=device)
    w_sim = R_sim.matmul(wS.unsqueeze(-1)).squeeze(-1)
    R_sim = R_sim.matmul(RBS.transpose(1, 2))
    return a_sim, w_sim, R_sim


def _npz_framerate(cdata):
    if "mocap_framerate" in cdata:
        return int(cdata["mocap_framerate"])
    if "mocap_frame_rate" in cdata:
        return int(cdata["mocap_frame_rate"])
    raise ValueError("missing AMASS mocap framerate")


def _gender(cdata):
    value = str(cdata["gender"]) if "gender" in cdata else "unknown"
    if value == "b'female'":
        return "female"
    if value == "b'male'":
        return "male"
    return value


def _shape(cdata):
    if "betas" not in cdata:
        return torch.zeros(10)
    betas = cdata["betas"][:10].astype(np.float32)
    if betas.shape[0] < 10:
        betas = np.pad(betas, (0, 10 - betas.shape[0]))
    return torch.from_numpy(betas).float()


def _resample_pose_tran_to_60fps(cdata, seq_name):
    framerate = _npz_framerate(cdata)
    poses = cdata["poses"]
    trans = cdata["trans"]
    if poses.shape[0] < framerate * 0.5:
        raise ValueError(f"{seq_name}: too short")

    if framerate == 120:
        pose = torch.from_numpy(poses[::2].astype(np.float32)).view(-1, poses.shape[-1] // 3, 3)[:, :24]
        tran = torch.from_numpy(trans[::2].astype(np.float32)).view(-1, 3)
    elif framerate in (60, 59):
        pose = torch.from_numpy(poses.astype(np.float32)).view(-1, poses.shape[-1] // 3, 3)[:, :24]
        tran = torch.from_numpy(trans.astype(np.float32)).view(-1, 3)
    else:
        origin_pose = poses.reshape(-1, poses.shape[-1] // 3, 3)[:, :24]
        origin_tran = trans.reshape(-1, 3)
        origin_t = np.arange(origin_pose.shape[0]) / framerate
        target_t = np.arange(0, origin_t[-1], 1 / 60)
        if len(target_t) < 4:
            raise ValueError(f"{seq_name}: too short after 60 FPS resampling")

        pose_np = np.empty((len(target_t), 24, 3), dtype=np.float32)
        for joint_idx in range(24):
            pose_np[:, joint_idx] = Slerp(origin_t, Rotation.from_rotvec(origin_pose[:, joint_idx]))(
                target_t
            ).as_rotvec()
        tran_np = interp1d(origin_t, origin_tran, axis=0)(target_t).astype(np.float32)
        pose = torch.from_numpy(pose_np)
        tran = torch.from_numpy(tran_np).view(-1, 3)

    pose = pose.contiguous().view(-1, 72)
    if pose.shape[0] < 4:
        raise ValueError(f"{seq_name}: too short for IMU finite differences")

    root_rot = art.math.axis_angle_to_rotation_matrix(pose[:, :3])
    pose[:, :3] = art.math.rotation_matrix_to_axis_angle(AMASS_TO_SMPL_FRAME.matmul(root_rot))
    tran = AMASS_TO_SMPL_FRAME.matmul(tran.unsqueeze(-1)).view_as(tran)
    return pose.float(), tran.float()


def _iter_amass_files(raw_amass_dir, datasets):
    seen = set()
    for dataset in datasets:
        aliases = AMASS_DATASET_ALIASES.get(dataset, (dataset,))
        for dirname in aliases:
            patterns = [
                os.path.join(raw_amass_dir, dirname, dirname, "*", "*_poses.npz"),
                os.path.join(raw_amass_dir, dirname, dirname, "*", "*_stageii.npz"),
                os.path.join(raw_amass_dir, dirname, "*", "*_poses.npz"),
                os.path.join(raw_amass_dir, dirname, "*", "*_stageii.npz"),
            ]
            for pattern in patterns:
                for npz_path in sorted(glob.glob(pattern)):
                    if npz_path in seen:
                        continue
                    seen.add(npz_path)
                    yield dataset, npz_path


def _to_raw_style_fields(aM, wM, RMB):
    """Create fields compatible with `test.py`'s real-IMU conversion path."""
    gravity = GRAVITY.to(aM.device)
    magnetic = MAGNETIC_FIELD.to(aM.device)
    aS = RMB.transpose(-1, -2).matmul((aM - gravity).unsqueeze(-1)).squeeze(-1)
    wS = RMB.transpose(-1, -2).matmul(wM.unsqueeze(-1)).squeeze(-1)
    mS = RMB.transpose(-1, -2).matmul(magnetic.view(1, 1, 3, 1).expand_as(RMB[..., :1])).squeeze(-1)
    eye = torch.eye(3).repeat(6, 1, 1)
    return {
        "RIM": eye,
        "RSB": eye.clone(),
        "RIS": RMB.cpu(),
        "aS": aS.cpu(),
        "wS": wS.cpu(),
        "mS": mS.cpu(),
    }


@torch.no_grad()
def synthesize_sequence(pose, tran, use_eskf=False):
    grot, joint, vert = _forward_smpl(pose, tran)
    p_imu = vert
    R_imu_body = grot[:, IMU_JOINTS]
    aM, wM, RMB = _syn_imu(p_imu, R_imu_body, use_eskf=use_eskf)
    raw_fields = _to_raw_style_fields(aM, wM, RMB)
    raw_fields.update(
        {
            "aM": aM.cpu(),
            "wM": wM.cpu(),
            "RMB": RMB.cpu(),
            "joint": joint.cpu(),
            "v_imu": vert.cpu(),
        }
    )
    return raw_fields


def process_amass(args):
    datasets = args.datasets or AMASS_DATASETS

    def new_data_dict():
        return {
            "name": [],
            "RIM": [],
            "RSB": [],
            "RIS": [],
            "aS": [],
            "wS": [],
            "mS": [],
            "aM": [],
            "wM": [],
            "RMB": [],
            "tran": [],
            "pose": [],
            "joint": [],
            "v_imu": [],
            "gender": [],
            "shape": [],
        }

    def save_shard(data, shard_idx):
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shard_path = output.with_name(f"{output.stem}_shard{shard_idx:05d}{output.suffix}")
        torch.save(data, shard_path)
        num_frames = int(sum(_.shape[0] for _ in data["pose"]))
        print(f"Saved shard {shard_idx} with {len(data['pose'])} sequences / {num_frames} frames to {shard_path}")
        return {
            "path": str(shard_path),
            "num_sequences": len(data["pose"]),
            "num_frames": num_frames,
            "names": list(data["name"]),
        }

    data = new_data_dict()
    manifest = {
        "output_stem": str(Path(args.output)),
        "raw_amass_dir": args.raw_amass_dir,
        "datasets": list(datasets),
        "seed": args.seed,
        "use_eskf": bool(args.use_eskf),
        "max_sequences": args.max_sequences,
        "max_frames": args.max_frames,
        "min_frames": args.min_frames,
        "shard_size": args.shard_size,
        "shards": [],
    }

    n_processed = 0
    n_failed = 0
    shard_idx = 0
    for dataset, npz_path in _iter_amass_files(args.raw_amass_dir, datasets):
        seq_name = npz_path[npz_path.rfind(dataset) : -4]
        try:
            cdata = np.load(npz_path, allow_pickle=True)
            pose, tran = _resample_pose_tran_to_60fps(cdata, seq_name)
            if args.min_frames and pose.shape[0] < args.min_frames:
                raise ValueError(f"{seq_name}: shorter than --min-frames={args.min_frames}")
            if args.max_frames:
                pose = pose[: args.max_frames]
                tran = tran[: args.max_frames]
            syn = synthesize_sequence(pose, tran, use_eskf=args.use_eskf)
        except Exception as exc:
            n_failed += 1
            print(f"Fail to process {seq_name}: {exc}")
            continue

        data["name"].append(seq_name)
        data["pose"].append(pose.cpu())
        data["tran"].append(tran.cpu())
        data["gender"].append(_gender(cdata))
        data["shape"].append(_shape(cdata))
        for key in ("RIM", "RSB", "RIS", "aS", "wS", "mS", "aM", "wM", "RMB", "joint", "v_imu"):
            data[key].append(syn[key])

        n_processed += 1
        print(f"Finish Processing {seq_name}: n_frames {pose.shape[0]}")
        if args.shard_size and len(data["pose"]) >= args.shard_size:
            manifest["shards"].append(save_shard(data, shard_idx))
            shard_idx += 1
            data = new_data_dict()
        if args.max_sequences and n_processed >= args.max_sequences:
            break

    if n_processed == 0:
        raise RuntimeError("No AMASS sequence was processed. Check --raw-amass-dir and --datasets.")

    output = Path(args.output)
    if args.shard_size:
        if data["pose"]:
            manifest["shards"].append(save_shard(data, shard_idx))
        manifest["num_sequences"] = n_processed
        manifest["num_failed"] = n_failed
        manifest["num_frames"] = int(sum(_["num_frames"] for _ in manifest["shards"]))
        manifest_path = output.with_name(f"{output.stem}_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved manifest for {n_processed} sequences to {manifest_path}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output)
        print(f"Saved {n_processed} sequences to {output}")
    if n_failed:
        print(f"Skipped {n_failed} failed sequences")


def synthetic_smoke_test():
    pose = torch.zeros(16, 72)
    tran = torch.zeros(16, 3)
    tran[:, 0] = torch.linspace(0, 0.15, 16)
    syn = synthesize_sequence(pose, tran)
    print("smoke pose:", tuple(pose.shape))
    for key in ("aS", "wS", "mS", "RIS", "aM", "wM", "RMB", "joint", "v_imu"):
        print(f"smoke {key}:", tuple(syn[key].shape), "finite=", torch.isfinite(syn[key]).all().item())
    RMB = syn["RIM"].transpose(1, 2).matmul(syn["RIS"]).matmul(syn["RSB"])
    aM = syn["RIM"].transpose(1, 2).matmul(syn["RIS"]).matmul(syn["aS"].unsqueeze(-1)).squeeze(-1) + GRAVITY
    wM = syn["RIM"].transpose(1, 2).matmul(syn["RIS"]).matmul(syn["wS"].unsqueeze(-1)).squeeze(-1)
    print("smoke raw->model maxerr aM:", (aM - syn["aM"]).abs().max().item())
    print("smoke raw->model maxerr wM:", (wM - syn["wM"]).abs().max().item())
    print("smoke raw->model maxerr RMB:", (RMB - syn["RMB"]).abs().max().item())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-amass-dir", default="data/dataset_raw/AMASS", help="Root directory of raw AMASS.")
    parser.add_argument("--output", default="data/dataset_work/AMASS/globalpose_synth.pt")
    parser.add_argument("--datasets", nargs="*", default=None, help="AMASS subset names. Defaults to GlobalPose/PNP list.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Optional cap for smoke/bounded processing.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional per-sequence frame cap.")
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=0, help="Save every N processed sequences as one shard instead of one large .pt file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-eskf", action="store_true", help="Use slow C++ ESKF instead of GlobalPose's default fast integration.")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("device:", device)
    if args.synthetic_smoke_test:
        synthetic_smoke_test()
    else:
        process_amass(args)
