from scripts.analyze_instruction_target_definitions import bucket_landmarks, cue_flags


def test_landmark_buckets():
    assert [bucket_landmarks(value) for value in (0, 1, 2, 3, 9)] == [
        "0", "1", "2", "3+", "3+"
    ]


def test_overlapping_definition_cues():
    flags = cue_flags(
        "The small red building between the bank and church, across from Main Road",
        landmark_count=3,
        surroundings=["car park"],
    )
    assert flags["appearance"]
    assert flags["between"]
    assert flags["across_opposite"]
    assert flags["road_reference"]
    assert flags["named_landmark_relation"]
    assert flags["surrounding_context"]
