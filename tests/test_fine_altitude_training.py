from multiagent.active_altitude.fine_refinement import (
    FineAltitudeConfig,
    oracle_training_altitude,
)


def test_oracle_training_keeps_coarse_navigation_at_canonical_altitude():
    config = FineAltitudeConfig(altitude_levels_m=(50.0, 40.0), enter_distance_m=40.0)
    assert oracle_training_altitude(False, 5.0, config) == 50.0


def test_oracle_training_descends_only_inside_true_fine_radius():
    config = FineAltitudeConfig(altitude_levels_m=(50.0, 40.0), enter_distance_m=40.0)
    assert oracle_training_altitude(True, 40.0, config) == 40.0
    assert oracle_training_altitude(True, 40.01, config) == 50.0


def test_privileged_distance_is_guarded_from_evaluation_path():
    source = open('multiagent/agent.py').read()
    assert "getattr(self.args, 'train_fine_altitude', False) and train_ml is not None" in source
    assert 'true_distance_m = poses[i].xy.dist_to' in source
    assert 'if oracle_fine_altitude_enabled and not ended[i]' in source
