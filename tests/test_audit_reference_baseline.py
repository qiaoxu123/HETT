import gzip
import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts/audit_reference_baseline.py"
SPEC = importlib.util.spec_from_file_location("audit_reference_baseline", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_curve_ignores_repeated_best_block(tmp_path):
    path = tmp_path / "train.log.gz"
    text = """IL_loss 7.0 direction_loss 6.0 progress_loss 0.2 goal_predict_loss 0.1
epoch 0
val_seen , sr: 10.0, ne: 50.0
val_unseen , sr: 11.0, ne: 51.0
epoch 9
val_seen , sr: 99.0, ne: 1.0
val_unseen , sr: 99.0, ne: 1.0
IL_loss 6.0 direction_loss 5.0 progress_loss 0.1 goal_predict_loss 0.05
epoch 1
val_seen , sr: 12.0, ne: 48.0
val_unseen , sr: 13.0, ne: 49.0
"""
    with gzip.open(path, "wt") as stream:
        stream.write(text)
    rows = MODULE.training_curve(path)
    assert [row["epoch"] for row in rows] == [0, 1]
    assert rows[0]["val_unseen"]["sr"] == 11.0


def test_live_epoch_indices_align_one_based_telemetry_with_legacy_logs():
    assert MODULE.live_epoch_indices([{"epoch": 1}, {"epoch": 2}]) == [0, 1]


def test_live_epoch_indices_reject_gaps_or_zero_based_input():
    try:
        MODULE.live_epoch_indices([{"epoch": 0}])
    except ValueError as error:
        assert "one-based" in str(error)
    else:
        raise AssertionError("zero-based structured telemetry was silently accepted")
