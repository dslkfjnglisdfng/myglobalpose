import math
import sys
from pathlib import Path

import torch

EXTRA = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(EXTRA), str(ROOT)]

from gp_w_input_swap import CausalRMBWorldAngularVelocity, causal_w_sequence


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_static_rmb():
    r = torch.eye(3, dtype=torch.float64).repeat(8, 6, 1, 1)
    assert torch.equal(causal_w_sequence(r), torch.zeros(8, 6, 3, dtype=torch.float64))


def test_constant_world_angular_velocity():
    omega = 0.75
    r = torch.stack([rz(omega * t / 60.0).repeat(6, 1, 1) for t in range(10)])
    w = causal_w_sequence(r)
    assert torch.allclose(w[:2], torch.zeros_like(w[:2]))
    assert torch.allclose(w[2:, :, 2], torch.full_like(w[2:, :, 2], omega), atol=1e-10)
    assert torch.allclose(w[2:, :, :2], torch.zeros_like(w[2:, :, :2]), atol=1e-10)


def test_ema_hand_calculation():
    angles = [0.0, 0.0, 0.02, 0.06]
    r = torch.stack([rz(a).repeat(6, 1, 1) for a in angles])
    w = causal_w_sequence(r)
    raw2 = 0.02 / (2 / 60)
    raw3 = 0.06 / (2 / 60)
    assert abs(w[2, 0, 2].item() - raw2) < 1e-6
    assert abs(w[3, 0, 2].item() - (0.7 * raw2 + 0.3 * raw3)) < 1e-6


def test_causality():
    r = torch.stack([rz(0.01 * t * t).repeat(6, 1, 1) for t in range(12)])
    changed = r.clone()
    changed[8:] = torch.stack([rz(1.0 + 0.2 * t).repeat(6, 1, 1) for t in range(4)])
    assert torch.equal(causal_w_sequence(r)[:8], causal_w_sequence(changed)[:8])


def test_sequence_step_equivalence():
    r = torch.stack([rz(0.005 * t * t).repeat(6, 1, 1) for t in range(20)])
    state = CausalRMBWorldAngularVelocity()
    stepped = torch.stack([state.step(frame) for frame in r])
    assert (stepped - causal_w_sequence(r)).abs().max().item() < 1e-6


if __name__ == "__main__":
    for test in (
        test_static_rmb,
        test_constant_world_angular_velocity,
        test_ema_hand_calculation,
        test_causality,
        test_sequence_step_equivalence,
    ):
        test()
        print(f"PASS {test.__name__}")
