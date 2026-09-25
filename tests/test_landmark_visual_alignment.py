import torch

from scripts.audit_landmark_visual_alignment import retrieval_metrics


def test_retrieval_metrics_restricts_candidates_to_same_map():
    similarity = torch.tensor([
        [0.8, 0.1, 0.9],
        [0.2, 0.7, 0.9],
        [0.9, 0.8, 0.6],
    ])
    labels = ["church", "school", "tower"]
    maps = ["a", "a", "b"]

    result = retrieval_metrics(similarity, labels, maps, labels, maps)

    assert result["queries"] == 3
    assert result["top1"] == 100.0
