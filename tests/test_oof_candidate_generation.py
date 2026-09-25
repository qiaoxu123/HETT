from scripts.generate_oof_geometry_candidates import balanced_map_folds


def test_balanced_map_folds_never_split_a_map_across_folds():
    rows = [
        {"map_name": "a"}, {"map_name": "a"}, {"map_name": "a"},
        {"map_name": "b"}, {"map_name": "b"},
        {"map_name": "c"}, {"map_name": "d"},
    ]

    folds, sizes = balanced_map_folds(rows, 2)

    assert set(folds[0]).isdisjoint(folds[1])
    assert set(folds[0]) | set(folds[1]) == {"a", "b", "c", "d"}
    assert sum(sizes) == len(rows)
