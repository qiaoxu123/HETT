import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'multiagent'))
from agent import configure_goal_head_only  # noqa: E402


class TinyVLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.decoder_2_action_full = nn.Linear(3, 2)
        self.decoder_2_progress_full = nn.Linear(3, 1)
        self.decoder_2_goal_full = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def test_only_xy_goal_decoder_remains_trainable():
    language, vision, vln = nn.Linear(3, 3), nn.Linear(3, 3), TinyVLN()
    configure_goal_head_only(language, vision, vln)
    assert not any(p.requires_grad for p in language.parameters())
    assert not any(p.requires_grad for p in vision.parameters())
    trainable = {name for name, parameter in vln.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith('decoder_2_goal_full.') for name in trainable)
    assert not any(p.requires_grad for p in vln.encoder.parameters())
    assert not any(p.requires_grad for p in vln.decoder_2_action_full.parameters())
    assert not any(p.requires_grad for p in vln.decoder_2_progress_full.parameters())
