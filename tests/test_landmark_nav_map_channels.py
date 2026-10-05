import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multiagent.maps.landmark_nav_map import LandmarkNavMap


class _FakeTrackingMap:
    def __init__(self, *_args, **_kwargs):
        self._value = np.zeros((2, 8, 8), dtype=np.float32)

    def mark_current_view_area(self, _pose):
        return self

    def to_array(self, dtype=np.float32):
        return self._value.astype(dtype)


class _FakeLandmarkMap:
    calls = []

    def __init__(self, _map_name, map_shape, _pixels_per_meter, names, contour_only=False):
        self.calls.append((names, contour_only))
        self._value = np.ones((1, *map_shape), dtype=np.float32)

    def to_array(self, dtype=np.float32):
        return self._value.astype(dtype)


class LandmarkNavMapHelperTest(unittest.TestCase):
    def test_episode_helper_uses_contour_only_for_global_landmarks(self):
        _FakeLandmarkMap.calls = []
        episode = SimpleNamespace(
            map_name="dummy",
            target_processed_description=SimpleNamespace(landmarks=["church"]),
            sample_trajectory=lambda _interval, _end: [object(), object()],
        )
        with patch(
            "multiagent.maps.landmark_nav_map.TrackingMap", _FakeTrackingMap
        ), patch(
            "multiagent.maps.landmark_nav_map.LandmarkMap", _FakeLandmarkMap
        ):
            maps = LandmarkNavMap.generate_maps_for_an_episode(
                episode,
                map_shape=(8, 8),
                pixels_per_meter=1.0,
                update_interval=1,
            )

        self.assertEqual(tuple(maps.shape), (2, 4, 8, 8))
        self.assertIn((None, True), _FakeLandmarkMap.calls)
        self.assertIn((["church"], False), _FakeLandmarkMap.calls)


if __name__ == "__main__":
    unittest.main()
