import json
from pathlib import Path

import torch

import articulate as art
from pl_va_state import (
    CausalButterworthLowpass,
    CausalRMBWorldAngularVelocityEMA,
    PLVAStateV1,
    causal_world_angular_velocity_from_rmb_sequence,
    causal_butterworth_lowpass_sequence,
    partial_initialize_from_official,
    pl_va_feature_sequence,
    world_angular_velocity_to_root_frame,
)
from pl_va_state_data import LengthBucketBatchSampler, collate_sequences
from pl_va_state_lightning import PLVAStateLightning, compute_losses


def test_rmb_static_first_and_causal():
    r = torch.eye(3).repeat(12, 6, 1, 1)
    out = causal_world_angular_velocity_from_rmb_sequence(r)
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(out, torch.zeros_like(out))
    changed = r.clone()
    changed[8:] = art.math.axis_angle_to_rotation_matrix(torch.tensor([[0.3, 0.0, 0.0]])).expand(4, 6, 3, 3)
    assert torch.equal(out[:8], causal_world_angular_velocity_from_rmb_sequence(changed)[:8])


def test_constant_world_angular_velocity_sequence_step_and_reference_alignment():
    dt, omega = 1 / 60, torch.tensor([0.0, 1.2, 0.0])
    aa = torch.arange(20)[:, None, None] * dt * omega
    r = art.math.axis_angle_to_rotation_matrix(aa.expand(-1, 6, -1).reshape(-1, 3)).reshape(20, 6, 3, 3)
    out = causal_world_angular_velocity_from_rmb_sequence(r, dt=dt)
    assert torch.equal(out[:2], torch.zeros_like(out[:2]))
    assert torch.allclose(out[2:], omega.expand_as(out[2:]), atol=2e-5)

    state = CausalRMBWorldAngularVelocityEMA(dt=dt)
    stepped = torch.stack([state.step(frame) for frame in r])
    assert (out - stepped).abs().max() < 1e-6

    delta = r[2:].matmul(r[:-2].transpose(-1, -2))
    raw = art.math.rotation_matrix_to_axis_angle(delta.reshape(-1, 3, 3)).reshape(18, 6, 3) / (2 * dt)
    reference = torch.zeros_like(out)
    reference[2] = raw[0]
    for t in range(3, len(r)):
        reference[t] = 0.7 * reference[t - 1] + 0.3 * raw[t - 2]
    assert torch.allclose(out, reference, atol=1e-7)


def test_ema_initialization_and_updates():
    dt = 1 / 60
    increments = torch.tensor([0.0, 0.02, 0.05, 0.01, 0.04, 0.03])
    angles = increments.cumsum(0)
    aa = torch.zeros(len(angles), 6, 3)
    aa[..., 2] = angles[:, None]
    r = art.math.axis_angle_to_rotation_matrix(aa.reshape(-1, 3)).reshape(len(angles), 6, 3, 3)
    out = causal_world_angular_velocity_from_rmb_sequence(r, dt=dt)
    delta = r[2:].matmul(r[:-2].transpose(-1, -2))
    raw = art.math.rotation_matrix_to_axis_angle(delta.reshape(-1, 3, 3)).reshape(-1, 6, 3) / (2 * dt)
    assert torch.allclose(out[2], raw[0], atol=1e-6)
    assert torch.allclose(out[3], 0.7 * out[2] + 0.3 * raw[1], atol=1e-6)
    assert torch.allclose(out[4], 0.7 * out[3] + 0.3 * raw[2], atol=1e-6)


def test_world_to_root_frame_contract():
    root = art.math.axis_angle_to_rotation_matrix(torch.tensor([[0.0, 0.0, 0.7]]))[0]
    w_m = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]])
    assert torch.allclose(world_angular_velocity_to_root_frame(w_m, root), w_m @ root)


def test_lightning_data_and_loss_contract():
    """The readable Lightning path preserves full sequences, masks, and losses."""
    torch.manual_seed(7)
    records = []
    for length in (6, 4):
        gravity = torch.randn(length, 3)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        records.append({
            "length": length,
            "feature": torch.randn(length, 102),
            "p_gt": torch.randn(length, 15),
            "v_gt": torch.randn(length, 15),
            "a_gt": torch.randn(length, 15),
            "g_gt": gravity,
            "init_legacy": torch.randn(18),
        })

    batch = collate_sequences(records)
    assert batch["feature"].shape == (2, 6, 102)
    assert batch["mask"].sum().item() == 10
    assert list(LengthBucketBatchSampler([6, 4], 2, False)) == [[1, 0]]

    stats = {"v_mean": torch.zeros(15), "v_std": torch.ones(15),
             "a_mean": torch.zeros(15), "a_std": torch.ones(15)}
    module = PLVAStateLightning(stats)
    output = module(batch)
    total, raw, weighted = compute_losses(output, batch, module.stats)
    assert torch.isfinite(total)
    assert set(raw) == set(weighted) == {
        "p", "v_direct", "v_state", "a", "g", "consistency", "jerk"
    }
    total.backward()
    gradient = module.model.net.linear2.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_filter_and_feature_contract():
    torch.manual_seed(1)
    a = torch.randn(30, 6, 3)
    r = torch.eye(3).repeat(30, 6, 1, 1)
    seq = causal_butterworth_lowpass_sequence(a)
    filt = CausalButterworthLowpass((6, 3))
    step = torch.stack([filt.step(x) for x in a])
    assert torch.allclose(seq, step, atol=1e-6)
    assert pl_va_feature_sequence(a, r).shape == (30, 102)


def test_state_integrators():
    model = PLVAStateV1(beta=0.7)
    state = model.initial_state(torch.zeros(15))
    raw = torch.zeros(1, 33); raw[:, 30:33] = torch.tensor([0.0, -1.0, 0.0])
    _, nxt, _ = model._update(raw, state)
    assert torch.equal(nxt.p, state.p)
    state.v.fill_(2.0)
    raw[:, :15] = 2.0
    _, nxt, _ = model._update(raw, state)
    assert torch.allclose(nxt.p, torch.full_like(nxt.p, 2 / 60), atol=1e-7)
    model.beta = 0.0
    state = model.initial_state(torch.zeros(15))
    raw[:, :15] = 0.0
    raw[:, 15:30] = 3.0
    _, nxt, _ = model._update(raw, state)
    assert torch.allclose(nxt.v, torch.full_like(nxt.v, 0.025), atol=1e-7)
    assert torch.allclose(nxt.p, torch.full_like(nxt.p, 0.0002083333), atol=1e-7)


def test_sequence_step_chunk_and_initialization(tmp_path):
    torch.manual_seed(2)
    model = PLVAStateV1().eval()
    report = partial_initialize_from_official(model, "data/weights.pt", tmp_path / "initialization_report.json")
    assert report["partially_copied_keys"]
    assert json.loads((tmp_path / "initialization_report.json").read_text())["copied_keys"]
    x, init = torch.randn(17, 102), torch.randn(18)
    full = model.forward_sequence(x, init)
    model.reset_stream(init[:15], init[15:])
    stepped = torch.cat([model.step(row)["pl_t"] for row in x])
    assert full["raw_output"].shape == (17, 33)
    assert full["pl"].shape == (17, 18)
    assert torch.allclose(full["pl"], stepped, atol=2e-6)
    first = model.forward_sequence(x[:8], init)
    second = model.forward_sequence(x[8:], init, state=first["state"])
    assert torch.allclose(full["pl"], torch.cat((first["pl"], second["pl"])), atol=2e-6)
    assert torch.isfinite(full["pl"]).all()
