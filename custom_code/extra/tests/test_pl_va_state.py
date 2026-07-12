import json
from pathlib import Path

import torch

import articulate as art
from pl_va_state import (
    CausalButterworthLowpass,
    PLVAStateV1,
    causal_angular_velocity_from_rmb_sequence,
    causal_angular_velocity_from_rmb_step,
    causal_butterworth_lowpass_sequence,
    partial_initialize_from_official,
    pl_va_feature_sequence,
)


def test_rmb_static_first_and_causal():
    r = torch.eye(3).repeat(12, 6, 1, 1)
    out = causal_angular_velocity_from_rmb_sequence(r)
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(out, torch.zeros_like(out))
    changed = r.clone()
    changed[8:] = art.math.axis_angle_to_rotation_matrix(torch.tensor([[0.3, 0.0, 0.0]])).expand(4, 6, 3, 3)
    assert torch.equal(out[:8], causal_angular_velocity_from_rmb_sequence(changed)[:8])


def test_constant_angular_velocity_and_k2_alignment():
    dt, omega = 1 / 60, torch.tensor([0.0, 1.2, 0.0])
    aa = torch.arange(20)[:, None, None] * dt * omega
    r = art.math.axis_angle_to_rotation_matrix(aa.expand(-1, 6, -1).reshape(-1, 3)).reshape(20, 6, 3, 3)
    out = causal_angular_velocity_from_rmb_sequence(r, dt)
    assert torch.allclose(out[1:], omega.expand_as(out[1:]), atol=2e-5)
    rel = r[1:].transpose(-1, -2).matmul(r[:-1])
    expected = -art.math.rotation_matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(19, 6, 3) / dt
    assert torch.allclose(out[1:], expected, atol=1e-7)
    step, prev, first = causal_angular_velocity_from_rmb_step(r[0], None, dt)
    assert first and torch.equal(step, torch.zeros_like(step))
    step, _, first = causal_angular_velocity_from_rmb_step(r[1], prev, dt)
    assert not first and torch.allclose(step, out[1], atol=1e-7)


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
