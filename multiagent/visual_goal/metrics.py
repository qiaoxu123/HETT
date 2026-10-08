"""Retrieval, distance-trend, leakage and diagnostic metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


DISTANCE_BINS = ((80.0, float("inf"), ">80m"), (40.0, 80.0, "40-80m"),
                 (20.0, 40.0, "20-40m"), (0.0, 20.0, "0-20m"))


def _id_eq(a, b):
    return np.asarray([x == y for x, y in zip(a, b)], dtype=bool)


def retrieval_metrics(query_features, candidate_features, query_ids, candidate_ids,
                      query_maps, candidate_maps, candidate_positions=None,
                      query_distances=None, query_episode_ids=None):
    q = _normalize(query_features)
    c = _normalize(candidate_features)
    scores = q @ c.T
    query_ids, candidate_ids = np.asarray(query_ids), np.asarray(candidate_ids)
    query_maps, candidate_maps = np.asarray(query_maps), np.asarray(candidate_maps)
    unique_ids = np.unique(candidate_ids)
    grouped_scores = np.stack([scores[:, candidate_ids == scene].max(axis=1) for scene in unique_ids], axis=1)
    grouped_maps = np.asarray([candidate_maps[np.flatnonzero(candidate_ids == scene)[0]] for scene in unique_ids])
    scores = grouped_scores
    candidate_ids = unique_ids
    candidate_maps = grouped_maps
    positive = query_ids[:, None] == candidate_ids[None, :]
    present = positive.any(axis=1)
    result = {"queries": int(len(q)), "candidates": int(len(c))}
    for k in (1, 5, 10):
        k_eff = min(k, scores.shape[1])
        inds = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
        result[f"R@{k}"] = float(np.mean([positive[i, ix].any() for i, ix in enumerate(inds)])) if len(q) else None
    pos_score = np.where(positive, scores, -np.inf).max(axis=1)
    same_map_neg = (query_maps[:, None] == candidate_maps[None, :]) & ~positive
    map_neg_score = np.where(same_map_neg, scores, -np.inf).max(axis=1)
    has_neg = np.isfinite(map_neg_score) & present
    result["same_map_hard_negative_accuracy"] = float(np.mean(pos_score[has_neg] > map_neg_score[has_neg])) if has_neg.any() else None
    result["same_map_margin"] = float(np.mean(pos_score[has_neg] - map_neg_score[has_neg])) if has_neg.any() else None
    pair_scores = []
    pair_labels = []
    for i in range(len(q)):
        if not present[i]:
            continue
        pair_scores.extend(scores[i, positive[i]].tolist())
        pair_labels.extend([1] * int(positive[i].sum()))
        neg = same_map_neg[i]
        pair_scores.extend(scores[i, neg].tolist())
        pair_labels.extend([0] * int(neg.sum()))
    result["same_map_auroc"] = float(roc_auc_score(pair_labels, pair_scores)) if len(set(pair_labels)) == 2 else None
    if query_distances is not None:
        dist = np.asarray(query_distances, dtype=float)
        own_scores = pos_score
        result["similarity_by_distance"] = {}
        for lo, hi, name in DISTANCE_BINS:
            mask = (dist >= lo) & (dist < hi) & np.isfinite(own_scores)
            result["similarity_by_distance"][name] = {
                "n": int(mask.sum()), "mean": float(own_scores[mask].mean()) if mask.any() else None,
            }
        distance_corr = spearmanr(dist[present], own_scores[present]) if present.sum() > 2 else None
        result["distance_spearman_rho"] = float(distance_corr.statistic) if distance_corr is not None else None
        result["distance_spearman_pvalue"] = float(distance_corr.pvalue) if distance_corr is not None else None
        if query_episode_ids is not None:
            episodes = np.asarray(query_episode_ids)
            per_episode = []
            for eid in np.unique(episodes):
                emask = episodes == eid
                values = []
                for lo, hi, _ in DISTANCE_BINS:
                    b = emask & (dist >= lo) & (dist < hi)
                    values.append(float(own_scores[b].mean()) if b.any() else np.nan)
                values = np.asarray(values)
                good = np.isfinite(values)
                if good.sum() >= 2:
                    scale = float(np.std(values[good]))
                    per_episode.append((values - np.mean(values[good])) / max(scale, 1e-6))
            result["episode_normalized_curve"] = np.nanmean(per_episode, axis=0).tolist() if per_episode else None
    return result


def _normalize(features):
    x = np.asarray(features, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def bin_by_distance(distances):
    distances = np.asarray(distances, dtype=float)
    labels = np.full(len(distances), "invalid", dtype=object)
    for lo, hi, name in DISTANCE_BINS:
        labels[(distances >= lo) & (distances < hi)] = name
    return labels


def partial_distance_correlation(distances, similarity, altitude, brightness, maps):
    """OLS residual correlation after altitude, brightness and map fixed effects."""
    from sklearn.linear_model import LinearRegression
    d = np.asarray(distances, dtype=float)
    s = np.asarray(similarity, dtype=float)
    z = np.asarray(altitude, dtype=float)
    b = np.asarray(brightness, dtype=float)
    m = np.asarray(maps)
    valid = np.isfinite(d + s + z + b)
    d, s, z, b, m = d[valid], s[valid], z[valid], b[valid], m[valid]
    if len(d) < 5:
        return {"n": int(len(d)), "partial_distance_rho": None}
    onehot = np.eye(len(np.unique(m)))[np.unique(m, return_inverse=True)[1]]
    covariates = np.column_stack([z, b, onehot])
    rd = d - LinearRegression().fit(covariates, d).predict(covariates)
    rs = s - LinearRegression().fit(covariates, s).predict(covariates)
    corr = spearmanr(rd, rs)
    return {"n": int(len(d)), "partial_distance_rho": float(corr.statistic),
            "partial_distance_pvalue": float(corr.pvalue)}


def hard_negative_metrics(query_features, candidate_features, query_meta, candidate_meta, seed=7):
    """Positive-vs-four explicit negative pools; all candidate choice is post-encoder."""
    q = _normalize(query_features)
    c = _normalize(candidate_features)
    raw_sims = q @ c.T
    scene_ids = np.unique([x["scene_key"] for x in candidate_meta])
    sims = np.stack([raw_sims[:, [j for j, cm in enumerate(candidate_meta) if cm["scene_key"] == sid]].max(axis=1) for sid in scene_ids], axis=1)
    scene_meta = [next(cm for cm in candidate_meta if cm["scene_key"] == sid) for sid in scene_ids]
    rng = np.random.default_rng(seed)
    pools = {name: [] for name in ("random", "same_map", "nearby", "visually_similar")}
    records = []
    for i, qmeta in enumerate(query_meta):
        pos = np.asarray([j for j, cmeta in enumerate(scene_meta) if cmeta["scene_key"] == qmeta["scene_key"]])
        neg = np.asarray([j for j, cmeta in enumerate(scene_meta) if cmeta["scene_key"] != qmeta["scene_key"]])
        if not len(pos) or not len(neg):
            continue
        ps = float(sims[i, pos].max())
        same = np.asarray([j for j in neg if scene_meta[j]["map_name"] == qmeta["map_name"]])
        if len(same):
            same_scores = sims[i, same]
            nearby_mask = np.asarray([
                np.linalg.norm(np.asarray(scene_meta[j]["target_xy"]) - np.asarray(qmeta["target_xy"])) <= 200.0
                for j in same
            ])
            nearby = same[nearby_mask]
        else:
            nearby = np.asarray([], dtype=int)
        random_j = int(rng.choice(neg))
        visually_similar_j = int(neg[np.argmax(sims[i, neg])])
        choices = {
            "random": float(sims[i, random_j]),
            "same_map": float(sims[i, same].max()) if len(same) else np.nan,
            "nearby": float(sims[i, nearby].max()) if len(nearby) else np.nan,
            "visually_similar": float(sims[i, visually_similar_j]),
        }
        records.append((ps, choices))
    for name in pools:
        vals = [(p, n[name]) for p, n in records if np.isfinite(n[name])]
        positives = np.asarray([x[0] for x in vals])
        negatives = np.asarray([x[1] for x in vals])
        pool_scores = np.concatenate([positives, negatives])
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        pools[name] = {
            "n": int(len(vals)),
            "accuracy": float(np.mean(positives > negatives)) if len(vals) else None,
            "margin": float(np.mean(positives - negatives)) if len(vals) else None,
            "auroc": float(roc_auc_score(labels, pool_scores)) if len(vals) else None,
        }
    return pools
