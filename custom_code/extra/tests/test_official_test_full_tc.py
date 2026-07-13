import sys
from pathlib import Path

import torch


EXTRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTRA))

from run_official_test_full_tc import inventory
from summarize_official_test_full_tc import translation_windows


def test_inventory_keeps_every_sequence():
    data = {"pose": [torch.zeros(2, 72), torch.zeros(3, 72)], "name": ["s1_a", "s5_b"]}
    result = inventory(data, Path("release.pt"))
    assert result["sequence_count"] == 2
    assert result["total_frames"] == 5
    assert [row["name"] for row in result["sequences"]] == ["s1_a", "s5_b"]


def test_translation_windows_exact_simple_motion():
    target = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    pred = target.clone()
    result = translation_windows(pred, target)
    assert torch.equal(result["1"], torch.zeros_like(result["1"]))
    assert torch.equal(result["2"], torch.zeros_like(result["2"]))
    assert torch.equal(result["3"], torch.zeros_like(result["3"]))
    assert result["4"].numel() == 0
