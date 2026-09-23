import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_altitude_semantics.py"
SPEC = importlib.util.spec_from_file_location("semantic_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sample_key_matches_real_and_zoom_pair():
    real = MODULE.sample_key("/x/val_seen_00001_p002_60m.jpg", 60, "real")
    zoom = MODULE.sample_key("/x/val_seen_00001_p002_60m_zoom_from_120m.jpg", 60, "zoom")
    assert real == zoom == "val_seen_00001_p002"


def test_paired_bootstrap_delta_direction():
    result = MODULE.bootstrap_delta([0.5, 0.7], [0.2, 0.4], repetitions=1000)
    assert result["mean_delta_pp"] == pytest.approx(30.0)
    assert result["wins"] == 2
    assert result["losses"] == 0
