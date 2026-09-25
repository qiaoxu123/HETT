from scripts.build_temporal_belief_dataset import temporal_indices


def test_temporal_indices_use_only_sequence_length():
    assert temporal_indices(10, 3) == [0, 4, 9]
    assert temporal_indices(2, 3) == [0, 1]
    assert temporal_indices(0, 3) == []
