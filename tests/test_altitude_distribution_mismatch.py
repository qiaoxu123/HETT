from multiagent.active_altitude.fine_refinement import (
    FineAltitudeConfig,
    FineAltitudeController,
    FineAltitudeSignals,
    oracle_training_altitude,
)


def test_predicted_controller_can_descend_while_true_target_is_far():
    """Expose the rule mismatch without using a real validation trajectory."""
    config = FineAltitudeConfig(
        altitude_levels_m=(50.0, 40.0),
        enter_distance_m=40.0,
        stable_goal_window=2,
        transition_confirm_steps=2,
    )
    controller = FineAltitudeController(config)
    state = controller.initial_state()
    signals = FineAltitudeSignals(
        goal_probabilities=(0.99, 0.01),
        predicted_goal_xy=(0.25, 0.25),
        predicted_distance_m=10.0,
    )

    # A stable, confident, nearby *predicted* goal eventually triggers descent.
    controller.step(state, signals)
    controller.step(state, signals)
    predicted_decision = controller.step(state, signals)
    assert predicted_decision.target_altitude_m == 40.0

    # The training oracle sees that the true target is actually far away and
    # presents a 50 m observation for the same conceptual situation.
    training_altitude = oracle_training_altitude(
        stage1_ended=True,
        distance_to_target_m=100.0,
        config=config,
    )
    assert training_altitude == 50.0

