import numpy as np

from multiagent.visual_goal.metrics import retrieval_metrics, hard_negative_metrics
from multiagent.visual_goal.candidate_rerank import rerank_topk, candidate_recall, oracle_order


def test_retrieval_metrics_include_map_hard_negative_and_margin():
    q = np.asarray([[1., 0.], [0., 1.]])
    c = np.asarray([[0.9, .1], [.7, .3], [.1, .9]])
    m = retrieval_metrics(q, c, ["a", "b"], ["a", "x", "b"],
                          ["m", "m"], ["m", "m", "m"],
                          query_distances=[10, 50], query_episode_ids=["e1", "e2"])
    assert m["same_map_hard_negative_accuracy"] == 1.0
    assert m["same_map_margin"] > 0
    assert "R@5" in m and "distance_spearman_rho" in m


def test_hard_negative_controls_and_scene_dedup():
    q = np.asarray([[1., 0.]])
    c = np.asarray([[.9, .1], [.85, .15], [.8, .2], [0., 1.]])
    qmeta = [{"scene_key": "p", "map_name": "m", "target_xy": [0, 0]}]
    cmeta = [
        {"scene_key": "p", "map_name": "m", "target_xy": [0, 0]},
        {"scene_key": "p", "map_name": "m", "target_xy": [0, 0]},
        {"scene_key": "n", "map_name": "m", "target_xy": [10, 0]},
        {"scene_key": "r", "map_name": "other", "target_xy": [0, 0]},
    ]
    result = hard_negative_metrics(q, c, qmeta, cmeta)
    assert result["same_map"]["n"] == 1
    assert result["nearby"]["n"] == 1


def test_visual_rerank_and_oracle_are_candidate_order_equivariant():
    xy = np.asarray([[0, 0], [10, 0], [100, 0]], dtype=float)
    b, v, tgt = np.asarray([.5, .4, .1]), np.asarray([.1, .8, .2]), np.asarray([100, 0])
    r1 = rerank_topk(b, v, xy, tgt, .7)
    perm = np.asarray([2, 0, 1])
    r2 = rerank_topk(b[perm], v[perm], xy[perm], tgt, .7)
    assert np.allclose(xy[r1["top1_index"]], xy[perm][r2["top1_index"]])
    assert np.array_equal(oracle_order(xy, tgt)[:1], [2])
    assert candidate_recall(xy, tgt)["R@1/3"] is False

