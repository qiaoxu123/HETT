from __future__ import annotations

import numpy as np


def retrieval_metrics(similarity: np.ndarray, positive: np.ndarray, same_map: np.ndarray | None = None) -> dict:
    """Rows are queries; positive[i] is the matching template column."""
    s = np.asarray(similarity, dtype=float)
    positive = np.asarray(positive, dtype=int)
    if s.ndim != 2 or len(positive) != s.shape[0]:
        raise ValueError("expected QxN similarity and Q positive indices")
    ranks = np.argsort(-s, axis=1)
    rank = np.array([int(np.flatnonzero(row == pos)[0]) + 1 for row, pos in zip(ranks, positive)])
    pos_score = s[np.arange(len(positive)), positive]
    neg_score = np.empty(len(positive), dtype=float)
    for i, pos in enumerate(positive):
        allowed = np.ones(s.shape[1], dtype=bool)
        allowed[pos] = False
        if same_map is not None:
            allowed &= np.asarray(same_map[i], dtype=bool)
        neg_score[i] = np.max(s[i, allowed]) if allowed.any() else np.nan
    labels = np.zeros(s.size, dtype=int)
    labels[np.arange(len(positive)) * s.shape[1] + positive] = 1
    scores = s.ravel()
    # Mann–Whitney U with average ranks for ties; no sklearn runtime dependency.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]: end += 1
        ranks[order[begin:end]] = (begin + 1 + end) / 2.0
        begin = end
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    margins = pos_score - neg_score
    return {f"recall@{k}": float(np.mean(rank <= k)) for k in (1, 5, 10)} | {
        "auroc": auc, "positive_negative_margin": float(np.nanmean(margins)),
        "same_map_hard_negative_accuracy": float(np.nanmean(margins > 0)),
        "mean_positive_similarity": float(np.mean(pos_score)), "mean_hard_negative_similarity": float(np.nanmean(neg_score)),
        "n_queries": int(len(positive)), "ranks": rank.tolist(),
    }


def distance_correlation(distance: np.ndarray, similarity: np.ndarray) -> float:
    from scipy.stats import spearmanr
    result = spearmanr(distance, similarity, nan_policy="omit")
    return float(result.statistic)
