from multiagent.active_altitude.cache import ObservationLRU, quantize_observation
from multiagent.active_altitude.controller import AltitudeAction, NavigationPhase
from multiagent.active_altitude.teacher import build_teacher_targets


def test_teacher_blends_human_height_without_overriding_oracle_structure():
    target = build_teacher_targets(15, current_level=1, human_altitude_m=120)
    assert target.phase == NavigationPhase.REFINEMENT
    assert target.altitude_level_distribution[0] > target.altitude_level_distribution[3]
    assert target.altitude_action_distribution.sum() == 1.0
    assert target.altitude_action == AltitudeAction.DESCEND
    assert target.stop_target == 1.0


def test_observation_cache_is_quantized_and_bounded():
    cache = ObservationLRU[str](capacity=2)
    first = quantize_observation("map", 1, 1, 0, 3)
    same = quantize_observation("map", 2, 2, 0.01, 3)
    second = quantize_observation("map", 8, 1, 0, 3)
    third = quantize_observation("map", 16, 1, 0, 3)
    assert first == same
    cache.put(first, "first")
    cache.put(second, "second")
    cache.put(third, "third")
    assert len(cache) == 2
    assert cache.get(first) is None
