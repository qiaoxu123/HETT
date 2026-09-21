import sys
from pathlib import Path
from unittest.mock import patch

from multiagent.parser import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_test_unseen_requires_explicit_opt_in():
    with patch.object(sys, 'argv', ['test', '--mode', 'eval']):
        assert not parse_args().include_test_unseen
    with patch.object(sys, 'argv', ['test', '--mode', 'eval', '--include_test_unseen']):
        assert parse_args().include_test_unseen


def test_validation_builder_uses_opt_in_flag():
    source = (ROOT / 'multiagent/main.py').read_text()
    assert "val_env_names = ['val_seen', 'val_unseen']" in source
    assert 'if args.include_test_unseen:' in source
