#!/usr/bin/env python3
"""Select fixed-height observations by predicted candidate discrimination.

The selector never reads target coordinates.  Before each observation it uses
only candidate belief, geometric visibility/separation and travel distance.
RGB evidence is revealed only after the selected view has been chosen.
"""

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.projection import transform_points, view_corners, world_to_image_homography
from independent_nav.semantic import ProjectSemanticNet
from scripts.evaluate_target_free_candidate_views import render_observation
from scripts.evaluate_temporal_candidate_reranking import (
    episode_key,
    local_score,
    navigation_metrics,
    prepare_episode_data,
)


def pose_xy(raw_pose):
    return np.asarray(raw_pose[:2], dtype=np.float64)


def load_starts(data_root, split):
    rows = json.load(open(data_root / "processed_citynav" / f"citynav_{split}.json"))
    return {episode_key(row): pose_xy(row["trajectory"][0]) for row in rows}


def make_view_options(candidates, start):
    weights = np.exp(-np.arange(len(candidates), dtype=np.float64))
    weighted_center = (candidates * weights[:, None]).sum(0) / weights.sum()
    center = candidates.mean(0)
    return np.vstack((start[None], candidates, center[None], weighted_center[None]))


def view_quality(candidates, center, altitude):
    offset = np.abs(candidates - center[None]) / altitude
    edge = offset.max(axis=1)
    return np.clip(1.0 - edge, 0.0, 1.0)


def softmax(values, temperature=0.25):
    shifted = values / temperature
    shifted -= shifted.max()
    values = np.exp(shifted)
    return values / values.sum()


def choose_view(row, accumulated, seen_quality, current, selected,
                beta, separation_weight, travel_weight):
    count = len(row["candidates"])
    prior = -np.arange(count, dtype=np.float64) / max(count - 1, 1)
    belief = softmax(accumulated + beta * prior)
    uncertainty = -(belief * np.log(belief + 1e-12)
                    + (1.0 - belief) * np.log(1.0 - belief + 1e-12))
    pair_distance = np.linalg.norm(
        row["candidates_world"][:, None] - row["candidates_world"][None], axis=-1)
    pair_separation = np.clip(pair_distance / 20.0, 0.0, 1.0)

    best = None
    for index, (center, quality) in enumerate(zip(row["view_centers"], row["quality"])):
        if index in selected:
            continue
        marginal = quality * (1.0 - seen_quality)
        coverage = float(np.sum(uncertainty * marginal))
        pair = 0.0
        for left in range(count):
            for right in range(left + 1, count):
                pair += (belief[left] * belief[right]
                         * min(marginal[left], marginal[right])
                         * pair_separation[left, right])
        travel = float(np.linalg.norm(center - current) / 100.0)
        utility = coverage + separation_weight * pair - travel_weight * travel
        key = (utility, -travel, -index)
        if best is None or key > best[0]:
            best = key, index
    return best[1]


def simulate(row, evidence_field, beta, separation_weight, travel_weight, views):
    accumulated = np.zeros(len(row["candidates"]), dtype=np.float64)
    seen_quality = np.zeros_like(accumulated)
    current = row["start"]
    selected = []
    for _ in range(views):
        index = choose_view(
            row, accumulated, seen_quality, current, set(selected), beta,
            separation_weight, travel_weight)
        selected.append(index)
        accumulated = 1.0 - (1.0 - accumulated) * (1.0 - row[evidence_field][index])
        seen_quality = np.maximum(seen_quality, row["quality"][index])
        current = row["view_centers"][index]
    prior = -np.arange(len(accumulated), dtype=np.float64) / max(len(accumulated) - 1, 1)
    return int(np.argmax(accumulated + beta * prior)), selected


def simulate_rank_order(row, evidence_field, beta, views):
    # Option 0 is the start; options 1..5 are candidate ranks 1..5.
    selected = list(range(1, views + 1))
    evidence = row[evidence_field][selected]
    accumulated = 1.0 - np.prod(1.0 - evidence, axis=0)
    prior = -np.arange(len(accumulated), dtype=np.float64) / max(len(accumulated) - 1, 1)
    return int(np.argmax(accumulated + beta * prior)), selected


def simulate_challenger(row, evidence_field, beta, views):
    """Inspect rank 1, then the strongest not-yet-centered challenger."""
    count = len(row["candidates"])
    prior = -np.arange(count, dtype=np.float64) / max(count - 1, 1)
    accumulated = np.zeros(count, dtype=np.float64)
    selected_candidates = []
    selected_views = []
    for _ in range(views):
        if not selected_candidates:
            candidate = 0
        else:
            score = accumulated + beta * prior
            score[selected_candidates] = -np.inf
            candidate = int(np.argmax(score))
        view = candidate + 1  # Option 0 is the start view.
        selected_candidates.append(candidate)
        selected_views.append(view)
        accumulated = 1.0 - (1.0 - accumulated) * (1.0 - row[evidence_field][view])
    return int(np.argmax(accumulated + beta * prior)), selected_views


def ranks(rows, policy, evidence_field, beta, views,
          separation_weight=0.0, travel_weight=0.0):
    values, selections = [], []
    for row in rows:
        if policy == "information_gain":
            rank, selected = simulate(
                row, evidence_field, beta, separation_weight, travel_weight, views)
        elif policy == "challenger":
            rank, selected = simulate_challenger(
                row, evidence_field, beta, views)
        else:
            rank, selected = simulate_rank_order(row, evidence_field, beta, views)
        values.append(rank)
        selections.append(selected)
    return np.asarray(values), selections


def tune_selector(rows, beta, views):
    best = None
    for separation_weight in (0.0, 0.5, 1.0):
        for travel_weight in (0.0, 0.1, 0.25):
            rank, _ = ranks(
                rows, "information_gain", "correct_evidence", beta, views,
                separation_weight, travel_weight)
            value = navigation_metrics(rows, rank)
            key = (value["hit20"], -value["mean_ne_m"],
                   -separation_weight, -travel_weight)
            if best is None or key > best[0]:
                best = key, separation_weight, travel_weight
    return best[1], best[2]


def view_oracle_ranks(rows, beta, views, candidate_views_only=False):
    """Privileged diagnostic: best final candidate over all allowed view sets."""
    ranks_out = []
    for row in rows:
        count = len(row["candidates"])
        prior = -np.arange(count, dtype=np.float64) / max(count - 1, 1)
        options = range(1, 1 + count) if candidate_views_only else range(len(row["view_centers"]))
        best = None
        for selected in itertools.combinations(options, views):
            evidence = row["correct_evidence"][list(selected)]
            accumulated = 1.0 - np.prod(1.0 - evidence, axis=0)
            rank = int(np.argmax(accumulated + beta * prior))
            distance = float(np.linalg.norm(
                row["candidates"][rank] - row["target"]) * 410.0)
            key = (-float(distance <= 20.0), distance, selected)
            if best is None or key < best[0]:
                best = key, rank
        ranks_out.append(best[1])
    return np.asarray(ranks_out)


def selector_feature(row, view_index, altitude):
    relative = np.clip(
        (row["candidates_world"] - row["view_centers"][view_index]) / altitude,
        -2.0, 2.0).reshape(-1)
    center_from_start = np.clip(
        (row["view_centers"][view_index] - row["start"]) / 100.0, -4.0, 4.0)
    distances = np.linalg.norm(
        row["candidates_world"][:, None] - row["candidates_world"][None], axis=-1)
    pair = distances[np.triu_indices(len(distances), 1)] / 100.0
    view_kind = np.zeros(len(row["view_centers"]), dtype=np.float64)
    view_kind[view_index] = 1.0
    target_type = np.zeros(4, dtype=np.float64)
    target_type[row["predicted_type"]] = 1.0
    return np.concatenate((
        relative,
        row["quality"][view_index],
        center_from_start,
        [np.linalg.norm(center_from_start)],
        pair,
        view_kind,
        target_type,
    )).astype(np.float32)


def single_view_success(row, view_index, beta):
    count = len(row["candidates"])
    prior = -np.arange(count, dtype=np.float64) / max(count - 1, 1)
    rank = int(np.argmax(row["correct_evidence"][view_index] + beta * prior))
    distance = np.linalg.norm(row["candidates"][rank] - row["target"]) * 410.0
    return float(distance <= 20.0)


class ViewValueNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def object_fold(row):
    key = f"{row['episode_key'][0]}:{row['episode_key'][1]}".encode()
    return int(hashlib.sha1(key).hexdigest()[:8], 16) % 5


def train_view_value(rows, beta, altitude, device):
    features, labels, folds = [], [], []
    for row in rows:
        for view_index in range(len(row["view_centers"])):
            features.append(selector_feature(row, view_index, altitude))
            labels.append(single_view_success(row, view_index, beta))
            folds.append(object_fold(row))
    features = torch.from_numpy(np.stack(features))
    labels = torch.tensor(labels, dtype=torch.float32)
    folds = torch.tensor(folds)
    train = folds != 0
    valid = folds == 0
    model = ViewValueNet(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    positive = labels[train].sum().clamp_min(1.0)
    pos_weight = ((train.sum() - positive) / positive).to(device)
    best_loss, best_state = float("inf"), None
    generator = torch.Generator().manual_seed(0)
    for _ in range(80):
        model.train()
        order = torch.randperm(int(train.sum()), generator=generator)
        train_features = features[train][order]
        train_labels = labels[train][order]
        for offset in range(0, len(order), 512):
            x = train_features[offset:offset + 512].to(device)
            y = train_labels[offset:offset + 512].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(features[valid].to(device)), labels[valid].to(device),
                pos_weight=pos_weight).item()
        if loss < best_loss:
            best_loss = loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best_loss


@torch.inference_mode()
def learned_view_ranks(rows, model, beta, altitude, device):
    output, selections = [], []
    for row in rows:
        features = torch.from_numpy(np.stack([
            selector_feature(row, index, altitude)
            for index in range(len(row["view_centers"]))]))
        view = int(model(features.to(device)).argmax().item())
        count = len(row["candidates"])
        prior = -np.arange(count, dtype=np.float64) / max(count - 1, 1)
        rank = int(np.argmax(row["correct_evidence"][view] + beta * prior))
        output.append(rank)
        selections.append([view])
    return np.asarray(output), selections


@torch.inference_mode()
def collect(args, split, episode_data, model, device):
    starts = load_starts(args.data_root, split)
    grouped = defaultdict(list)
    for item in episode_data.values():
        grouped[item["episode_key"][0]].append(item)
    output = []
    for map_name, items in sorted(grouped.items()):
        rgb = cv2.imread(str(args.data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(map_name)
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            x_min, y_max = raster.bounds.left, raster.bounds.top
            for item in items:
                candidates_world = np.column_stack((
                    x_min + item["candidates"][:, 0] * 410.0,
                    y_max - item["candidates"][:, 1] * 410.0,
                ))
                start = starts[item["episode_key"]]
                centers = make_view_options(candidates_world, start)
                images, corners_by_view = [], []
                for center in centers:
                    image, corners = render_observation(
                        rgb, raster, center, args.altitude, args.render_size)
                    images.append(cv2.resize(
                        image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA))
                    corners_by_view.append(corners)
                tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
                probabilities = model(tensor.float().to(device).div(255.0)).softmax(1).cpu().numpy()
                semantic_class = item["predicted_type"] + 1
                wrong_class = ((item["predicted_type"] + 1) % 4) + 1
                radius_pixels = args.radius_m / (2.0 * args.altitude) * args.image_size
                correct, wrong, quality = [], [], []
                for view_index, center in enumerate(centers):
                    homography = world_to_image_homography(
                        corners_by_view[view_index], args.image_size)
                    pixels = transform_points(candidates_world, homography)
                    correct.append(np.asarray([
                        local_score(probabilities[view_index, semantic_class], point, radius_pixels)
                        for point in pixels]))
                    wrong.append(np.asarray([
                        local_score(probabilities[view_index, wrong_class], point, radius_pixels)
                        for point in pixels]))
                    quality.append(view_quality(candidates_world, center, args.altitude))
                output.append({
                    **item,
                    "start": start,
                    "candidates_world": candidates_world,
                    "view_centers": centers,
                    "quality": np.stack(quality),
                    "correct_evidence": np.stack(correct),
                    "wrong_evidence": np.stack(wrong),
                })
    return output


def selection_summary(selections):
    counts = defaultdict(int)
    for values in selections:
        for index in values:
            if index == 0:
                label = "start"
            elif 1 <= index <= 5:
                label = f"candidate_{index}"
            elif index == 6:
                label = "candidate_centroid"
            else:
                label = "prior_weighted_centroid"
            counts[label] += 1
    total = max(sum(counts.values()), 1)
    return {key: 100.0 * value / total for key, value in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--type-features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--radius-m", type=float, default=10.0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    episodes = prepare_episode_data(
        args.data_root, args.candidate_root, args.type_features, device)
    seen = collect(args, "val_seen", episodes["val_seen"], model, device)
    separation_weight, travel_weight = tune_selector(seen, args.beta, args.views)
    view_model, view_model_validation_loss = train_view_value(
        seen, args.beta, args.altitude, device)
    unseen = collect(args, "val_unseen", episodes["val_unseen"], model, device)

    zero = np.zeros(len(unseen), dtype=np.int64)
    rank_order, rank_order_views = ranks(
        unseen, "rank_order", "correct_evidence", args.beta, args.views)
    challenger, challenger_views = ranks(
        unseen, "challenger", "correct_evidence", args.beta, args.views)
    active, active_views = ranks(
        unseen, "information_gain", "correct_evidence", args.beta, args.views,
        separation_weight, travel_weight)
    wrong, _ = ranks(
        unseen, "information_gain", "wrong_evidence", args.beta, args.views,
        separation_weight, travel_weight)
    single_view_oracle = view_oracle_ranks(unseen, args.beta, 1)
    two_view_oracle = view_oracle_ranks(unseen, args.beta, args.views)
    candidate_two_view_oracle = view_oracle_ranks(
        unseen, args.beta, args.views, candidate_views_only=True)
    learned, learned_views = learned_view_ranks(
        unseen, view_model, args.beta, args.altitude, device)
    result = {
        "config": {
            "altitude_m": args.altitude,
            "views": args.views,
            "prior_weight_fixed_from_previous_val_seen_gate": args.beta,
            "separation_weight_selected_on_val_seen": separation_weight,
            "travel_weight_selected_on_val_seen": travel_weight,
            "view_options": "start, five predicted candidates, centroid, prior-weighted centroid",
        },
        "val_seen_episodes": len(seen),
        "val_unseen": {
            "episodes": len(unseen),
            "geometry_generator_top1": navigation_metrics(unseen, zero),
            "fixed_rank_order_two_views": navigation_metrics(unseen, rank_order),
            "adaptive_challenger_two_views": navigation_metrics(unseen, challenger),
            "information_gain_two_views": navigation_metrics(unseen, active),
            "learned_single_view_value": navigation_metrics(unseen, learned),
            "wrong_target_type_control": navigation_metrics(unseen, wrong),
            "privileged_single_view_oracle": navigation_metrics(
                unseen, single_view_oracle),
            "privileged_two_view_oracle": navigation_metrics(unseen, two_view_oracle),
            "privileged_candidate_centered_two_view_oracle": navigation_metrics(
                unseen, candidate_two_view_oracle),
            "view_selection_percent": selection_summary(active_views),
            "challenger_view_selection_percent": selection_summary(challenger_views),
            "fixed_view_selection_percent": selection_summary(rank_order_views),
            "learned_view_selection_percent": selection_summary(learned_views),
            "top5_oracle": navigation_metrics(unseen, np.asarray([
                np.linalg.norm(row["candidates"] - row["target"], axis=-1).argmin()
                for row in unseen])),
        },
        "anti_leakage": {
            "selector_inputs": "candidate belief, candidate geometry, start/current pose, fixed camera footprint",
            "rgb_timing": "RGB evidence is consumed only after a view is selected",
            "target_coordinates": "val_seen selector-weight selection and final metrics only",
            "target_mask_or_identity": False,
            "val_unseen_used_for_selection": False,
            "learned_selector_training": "val_seen targets only; object-grouped 4/1 train/selection split",
        },
        "learned_selector_validation_bce": view_model_validation_loss,
        "scope": "offline target-free next-view selection at fixed 50 m; dynamics excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
