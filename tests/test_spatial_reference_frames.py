import numpy as np

from scripts.audit_spatial_reference_frames import (
    TRANSFORMS,
    transform_accuracy,
)


def test_fixed_rotation_scoring_identifies_180_degree_frame():
    examples = [
        {"cue": "left", "centroid_delta": np.array([1.0, 0.2])},
        {"cue": "above", "centroid_delta": np.array([0.1, 1.0])},
    ]

    identity = transform_accuracy(examples, "centroid_delta", TRANSFORMS["identity"])
    rotated = transform_accuracy(examples, "centroid_delta", TRANSFORMS["rotate_180"])

    assert identity == 0.0
    assert rotated == 100.0
