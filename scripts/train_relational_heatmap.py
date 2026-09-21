"""Train and evaluate a landmark-geometry target heatmap without RGB or navigation."""

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from multiagent.mapdata import MAP_BOUNDS  # noqa: E402
from multiagent.models.relational_heatmap import RelationalHeatmap  # noqa: E402

SIZE = 64
MAX_LANDMARKS = 4
MAX_TOKENS = 80
SIGMA_METERS = 10.0
NMS_METERS = 20.0
ANGLE_NAMES = ["isotropic", "left", "right", "up", "down",
               "left_up", "right_up", "left_down", "right_down"]
DISTANCE_METERS = [0, 15, 30, 50, 80, 120]
TOKEN_RE = re.compile(r"<[^>]+>|[a-z0-9]+(?:'[a-z]+)?")
RELATIONS = {
    "between": re.compile(r"\b(between|middle of|in between)\b"),
    "left_right": re.compile(r"\b(left|right)(?:-hand)?\b"),
    "front_back": re.compile(r"\b(front|behind|back of|rear)\b"),
    "near": re.compile(r"\b(near|nearby|next to|adjacent|beside|close to|alongside)\b"),
    "cardinal": re.compile(r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\b"),
    "ordinal": re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|last|row|column)\b"),
}


def polygon_area(contour):
    points = np.asarray(contour, dtype=np.float64)
    return abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
               np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2


def relation_type(text):
    lower = text.lower()
    for name, pattern in RELATIONS.items():
        if pattern.search(lower):
            return name
    return "other"


def replace_names(text, names, selected=None):
    output = text.lower()
    indexed = sorted(enumerate(names), key=lambda item: len(item[1]), reverse=True)
    for index, name in indexed:
        marker = " <ref> " if selected == index else (" <landmark> " if selected is None else " <other> ")
        output = re.sub(re.escape(name.lower()), marker, output, flags=re.IGNORECASE)
    return output


def tokens(text):
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, texts, maximum=10000):
        counts = Counter(word for text in texts for word in tokens(text))
        words = [word for word, count in counts.most_common(maximum - 2) if count >= 2]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(word, 1) for word in tokens(text)][:MAX_TOKENS]
        return np.asarray(ids + [0] * (MAX_TOKENS - len(ids)), dtype=np.int64)


def load_objects():
    objects = json.load((ROOT / "data/cityrefer/objects.json").open())
    processed = json.load((ROOT / "data/cityrefer/processed_descriptions.json").open())
    lookups = {}
    for map_name, map_objects in objects.items():
        by_name = {}
        for object_id, obj in map_objects.items():
            name = obj.get("name", "")
            if not name:
                continue
            key = name.lower().strip()
            if key not in by_name or polygon_area(obj["contour"]) > polygon_area(by_name[key][1]["contour"]):
                by_name[key] = (int(object_id), obj)
        lookups[map_name] = by_name
    return objects, processed, lookups


def resolve_landmark(lookup, query):
    key = query.lower().strip()
    if key in lookup:
        return lookup[key]
    try:
        import Levenshtein
        return min(lookup.values(), key=lambda value: Levenshtein.distance(value[1]["name"], query))
    except ImportError:
        from difflib import SequenceMatcher
        return max(lookup.values(), key=lambda value: SequenceMatcher(None, value[1]["name"], query).ratio())


def load_split(split, objects, processed, lookups):
    records = json.load((ROOT / f"data/processed_citynav/citynav_{split}.json").open())
    samples = []
    truncated = 0
    for record in records:
        map_name = f"{record['area']}_block_{record['block']}"
        for object_id, annotation_id, description in zip(
                record["object_ids"], record["ann_ids"], record["descriptions"]):
            try:
                names = processed[map_name][str(object_id)][annotation_id]["landmarks"]
            except (KeyError, IndexError):
                names = []
            resolved = []
            seen = set()
            for query in names:
                landmark_id, landmark = resolve_landmark(lookups[map_name], query)
                if landmark_id not in seen:
                    resolved.append((landmark_id, landmark, query))
                    seen.add(landmark_id)
            if len(resolved) > MAX_LANDMARKS:
                truncated += 1
            resolved = resolved[:MAX_LANDMARKS]
            target = objects[map_name][str(object_id)]["position"][:2]
            samples.append({
                "map": map_name, "object_id": int(object_id), "annotation_id": int(annotation_id),
                "description": description, "target": target, "landmarks": resolved,
                "names": [item[2] for item in resolved], "relation": relation_type(description),
            })
    return samples, truncated


def map_mesh(map_name):
    bounds = MAP_BOUNDS[map_name]
    xs = np.linspace(bounds.x_min, bounds.x_max, SIZE, dtype=np.float32)
    ys = np.linspace(bounds.y_max, bounds.y_min, SIZE, dtype=np.float32)
    return np.meshgrid(xs, ys)


def landmark_basis(map_name, landmark):
    bounds = MAP_BOUNDS[map_name]
    x_grid, y_grid = map_mesh(map_name)
    points = np.asarray(landmark["contour"], dtype=np.float32)
    cols = (points[:, 0] - bounds.x_min) / (bounds.x_max - bounds.x_min) * (SIZE - 1)
    rows = (bounds.y_max - points[:, 1]) / (bounds.y_max - bounds.y_min) * (SIZE - 1)
    polygon = np.round(np.stack([cols, rows], axis=1)).astype(np.int32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    outside_cells = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    cell_m = 0.5 * ((bounds.x_max - bounds.x_min) + (bounds.y_max - bounds.y_min)) / (SIZE - 1)
    outside = outside_cells * cell_m
    cx, cy = landmark["position"][:2]
    dx, dy = x_grid - cx, y_grid - cy
    norm = np.sqrt(dx ** 2 + dy ** 2).clip(min=1e-3)
    unit_x, unit_y = dx / norm, dy / norm
    directions = [None, (-1, 0), (1, 0), (0, 1), (0, -1),
                  (-math.sqrt(.5), math.sqrt(.5)), (math.sqrt(.5), math.sqrt(.5)),
                  (-math.sqrt(.5), -math.sqrt(.5)), (math.sqrt(.5), -math.sqrt(.5))]
    angles = [np.ones_like(outside, dtype=np.float32)]
    for direction in directions[1:]:
        cosine = unit_x * direction[0] + unit_y * direction[1]
        angles.append(np.exp(3.0 * (cosine - 1.0)).astype(np.float32))
    sigmas = [10, 10, 12, 15, 20, 25]
    distances = [np.exp(-0.5 * ((outside - center) / sigma) ** 2).astype(np.float32)
                 for center, sigma in zip(DISTANCE_METERS, sigmas)]
    return np.stack(angles).astype(np.float16), np.stack(distances).astype(np.float16)


def pair_field(map_name, landmarks):
    field = np.zeros((SIZE, SIZE), dtype=np.float32)
    if len(landmarks) < 2:
        return field[None].astype(np.float16)
    x_grid, y_grid = map_mesh(map_name)
    p = np.stack([x_grid, y_grid], axis=-1)
    for left_index in range(len(landmarks)):
        for right_index in range(left_index + 1, len(landmarks)):
            a = np.asarray(landmarks[left_index][1]["position"][:2], dtype=np.float32)
            b = np.asarray(landmarks[right_index][1]["position"][:2], dtype=np.float32)
            vector = b - a
            denominator = max(float(np.dot(vector, vector)), 1e-6)
            t = np.clip(((p - a) * vector).sum(axis=-1) / denominator, 0, 1)
            projection = a + t[..., None] * vector
            distance = np.linalg.norm(p - projection, axis=-1)
            middle = 0.25 + 0.75 * np.sin(np.pi * t)
            field = np.maximum(field, np.exp(-0.5 * (distance / 20.0) ** 2) * middle)
    return field[None].astype(np.float16)


class HeatmapDataset(Dataset):
    def __init__(self, samples, vocab, basis_cache):
        self.samples = samples
        self.vocab = vocab
        self.basis_cache = basis_cache
        self.pair_cache = {}
        for sample in self.samples:
            names = sample["names"]
            sample["global_tokens"] = vocab.encode(replace_names(sample["description"], names))
            sample["ref_tokens"] = [
                vocab.encode(replace_names(sample["description"], names, selected=index))
                for index in range(len(names))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        angle_fields = np.zeros((MAX_LANDMARKS, len(ANGLE_NAMES), SIZE, SIZE), dtype=np.float16)
        distance_fields = np.zeros((MAX_LANDMARKS, len(DISTANCE_METERS), SIZE, SIZE), dtype=np.float16)
        ref_tokens = np.zeros((MAX_LANDMARKS, MAX_TOKENS), dtype=np.int64)
        valid = np.zeros(MAX_LANDMARKS, dtype=np.float32)
        for slot, (landmark_id, _, _) in enumerate(sample["landmarks"]):
            angle_fields[slot], distance_fields[slot] = self.basis_cache[(sample["map"], landmark_id)]
            ref_tokens[slot] = sample["ref_tokens"][slot]
            valid[slot] = 1
        bounds = MAP_BOUNDS[sample["map"]]
        target_col = (sample["target"][0] - bounds.x_min) / (bounds.x_max - bounds.x_min) * (SIZE - 1)
        target_row = (bounds.y_max - sample["target"][1]) / (bounds.y_max - bounds.y_min) * (SIZE - 1)
        pair_key = (sample["map"], tuple(item[0] for item in sample["landmarks"]))
        if pair_key not in self.pair_cache:
            self.pair_cache[pair_key] = pair_field(sample["map"], sample["landmarks"])
        return {
            "angle_fields": angle_fields, "distance_fields": distance_fields,
            "pair": self.pair_cache[pair_key],
            "ref_tokens": ref_tokens, "global_tokens": sample["global_tokens"], "valid": valid,
            "target_rc": np.asarray([target_row, target_col], dtype=np.float32),
            "cell_xy": np.asarray([(bounds.x_max - bounds.x_min) / (SIZE - 1),
                                   (bounds.y_max - bounds.y_min) / (SIZE - 1)], dtype=np.float32),
            "index": index,
        }


def collate(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key == "index":
            result[key] = values
        else:
            result[key] = torch.from_numpy(np.stack(values))
    return result


def target_fields(batch, device):
    rc = batch["target_rc"].to(device)
    cell = batch["cell_xy"].to(device)
    rows = torch.arange(SIZE, device=device)[None, :, None]
    cols = torch.arange(SIZE, device=device)[None, None, :]
    dy = (rows - rc[:, 0, None, None]) * cell[:, 1, None, None]
    dx = (cols - rc[:, 1, None, None]) * cell[:, 0, None, None]
    distance2 = dx.square() + dy.square()
    gaussian = torch.exp(-distance2 / (2 * SIGMA_METERS ** 2))
    gaussian = gaussian / gaussian.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return gaussian, distance2 <= 20.0 ** 2


def move(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def forward(model, batch):
    return model(batch["angle_fields"].float(), batch["distance_fields"].float(),
                 batch["pair"].float(), batch["ref_tokens"], batch["global_tokens"],
                 batch["valid"])


def nms_points(field, cell_xy, count=5):
    scores = field.copy()
    points = []
    row_grid, col_grid = np.ogrid[:SIZE, :SIZE]
    for _ in range(count):
        flat = int(np.argmax(scores))
        row, col = divmod(flat, SIZE)
        if not np.isfinite(scores[row, col]):
            break
        points.append((row, col))
        distance2 = ((row_grid - row) * cell_xy[1]) ** 2 + ((col_grid - col) * cell_xy[0]) ** 2
        scores[distance2 <= NMS_METERS ** 2] = -np.inf
    return points


def point_error(point, target_rc, cell_xy):
    return math.hypot((point[0] - target_rc[0]) * cell_xy[1],
                      (point[1] - target_rc[1]) * cell_xy[0])


def evaluate(model, loader, dataset, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            logits, relation_weights, pair_gate = forward(model, batch)
            probabilities = logits.flatten(1).softmax(dim=1).reshape(-1, SIZE, SIZE)
            _, within = target_fields(batch, device)
            masses = (probabilities * within).sum(dim=(1, 2)).cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            near = batch["distance_fields"][:, :, 0].amax(dim=1).cpu().numpy()
            weights = relation_weights.cpu().numpy()
            for offset, sample_index in enumerate(raw_batch["index"]):
                sample = dataset.samples[sample_index]
                target_rc = raw_batch["target_rc"][offset].numpy()
                cell_xy = raw_batch["cell_xy"][offset].numpy()
                learned_points = nms_points(probabilities[offset], cell_xy)
                near_points = nms_points(near[offset], cell_xy)
                centroid_points = []
                bounds = MAP_BOUNDS[sample["map"]]
                for _, landmark, _ in sample["landmarks"]:
                    x, y = landmark["position"][:2]
                    centroid_points.append(((bounds.y_max - y) / (bounds.y_max - bounds.y_min) * (SIZE - 1),
                                            (x - bounds.x_min) / (bounds.x_max - bounds.x_min) * (SIZE - 1)))
                def errors(points):
                    values = [point_error(point, target_rc, cell_xy) for point in points]
                    return values or [float("inf")]
                learned_errors, near_errors, centroid_errors = map(errors, [learned_points, near_points, centroid_points])
                outputs.append({
                    "index": sample_index, "map": sample["map"], "description": sample["description"],
                    "relation": sample["relation"], "target_rc": target_rc.tolist(),
                    "cell_xy": cell_xy.tolist(), "landmark_count": len(sample["landmarks"]),
                    "learned_top1_m": learned_errors[0], "learned_top5_m": min(learned_errors[:5]),
                    "near_top1_m": near_errors[0], "near_top5_m": min(near_errors[:5]),
                    "centroid_top1_m": centroid_errors[0], "centroid_top5_m": min(centroid_errors[:5]),
                    "mass20": float(masses[offset]), "points_rc": learned_points,
                    "relation_weights": weights[offset].tolist(), "pair_gate": float(pair_gate[offset]),
                    "heatmap": probabilities[offset].astype(np.float16),
                })
    return outputs


def metric(rows, prefix):
    top1 = np.asarray([row[f"{prefix}_top1_m"] for row in rows])
    top5 = np.asarray([row[f"{prefix}_top5_m"] for row in rows])
    finite = top1[np.isfinite(top1)]
    result = {
        "n": len(rows), "top1_hit20": round(100 * float(np.mean(top1 <= 20)), 2),
        "top5_recall20": round(100 * float(np.mean(top5 <= 20)), 2),
        "median_top1_m": round(float(np.median(finite)), 2) if len(finite) else None,
        "p90_top1_m": round(float(np.percentile(finite, 90)), 2) if len(finite) else None,
    }
    if prefix == "learned":
        result["mean_mass20"] = round(float(np.mean([row["mass20"] for row in rows])), 4)
    return result


def summarize(rows):
    summary = {name: metric(rows, name) for name in ["centroid", "near", "learned"]}
    summary["by_relation"] = {}
    for relation in sorted({row["relation"] for row in rows}):
        subset = [row for row in rows if row["relation"] == relation]
        summary["by_relation"][relation] = metric(subset, "learned")
    return summary


def write_report(path, result):
    lines = ["# 关系地标热图第一阶段验证", "",
             "输入仅含指令、具名地标名称及轮廓；未使用 RGB、目标轮廓或导航轨迹。", ""]
    for split in ["val_seen", "val_unseen"]:
        summary = result[split]
        lines += [f"## {split}", "", "| 方法 | Top-1 Hit@20 | Top-5 Recall@20 | 中位误差 | P90 |",
                  "|---|---:|---:|---:|---:|"]
        labels = {"centroid": "首个地标中心", "near": "无训练轮廓距离场", "learned": "关系热图模型"}
        for key in ["centroid", "near", "learned"]:
            row = summary[key]
            lines.append(f"| {labels[key]} | {row['top1_hit20']:.2f}% | {row['top5_recall20']:.2f}% | "
                         f"{row['median_top1_m']} m | {row['p90_top1_m']} m |")
        lines += ["", "### 关系热图分类型", "",
                  "| 类型 | N | Top-1 Hit@20 | Top-5 Recall@20 |", "|---|---:|---:|---:|"]
        for relation, row in summary["by_relation"].items():
            lines.append(f"| {relation} | {row['n']} | {row['top1_hit20']:.2f}% | {row['top5_recall20']:.2f}% |")
        lines.append("")
    unseen = result["val_unseen"]["learned"]
    gap = result["val_seen"]["learned"]["top1_hit20"] - unseen["top1_hit20"]
    passed = unseen["top5_recall20"] >= 70 and unseen["top1_hit20"] >= 35 and gap <= 15
    lines += ["## 预注册判定", "",
              f"- val_unseen Top-5：{unseen['top5_recall20']:.2f}%（门槛 70%）",
              f"- val_unseen Top-1：{unseen['top1_hit20']:.2f}%（门槛 35%）",
              f"- seen-unseen Top-1 差：{gap:+.2f} 点（上限 15 点）", "",
              "**通过。**" if passed else "**未通过。**", "",
              "这些是静态目标提议指标，不是导航成功率。"]
    path.write_text("\n".join(lines) + "\n")
    return passed


def plot_results(run_dir, history, result, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(history) + 1), [row["loss"] for row in history], marker="o")
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Training")
    methods = ["centroid", "near", "learned"]
    x = np.arange(len(methods)); width = 0.35
    for offset, split in enumerate(["val_seen", "val_unseen"]):
        values = [result[split][method]["top1_hit20"] for method in methods]
        axes[1].bar(x + (offset - .5) * width, values, width, label=split)
    axes[1].set_xticks(x, ["centroid", "distance", "learned"])
    axes[1].set(ylabel="Top-1 Hit@20 (%)", title="Target localization")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(run_dir / "metrics.png", dpi=180); plt.close(fig)

    chosen = []
    for relation in ["left_right", "front_back", "near", "between", "cardinal", "ordinal", "other"]:
        match = next((row for row in rows if row["relation"] == relation), None)
        if match:
            chosen.append(match)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = np.asarray(axes).reshape(-1)
    for axis, row in zip(axes, chosen):
        axis.imshow(row["heatmap"], cmap="magma", origin="upper")
        target = row["target_rc"]
        axis.scatter([target[1]], [target[0]], marker="x", c="cyan", s=60, label="GT")
        points = row["points_rc"]
        axis.scatter([point[1] for point in points], [point[0] for point in points],
                     facecolors="none", edgecolors="lime", s=45)
        axis.set_title(f"{row['relation']}  e={row['learned_top1_m']:.1f}m", fontsize=9)
        axis.axis("off")
    for axis in axes[len(chosen):]: axis.axis("off")
    fig.tight_layout(); fig.savefig(run_dir / "examples.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--train-limit", type=int, default=0, help="smoke-test only")
    parser.add_argument("--val-limit", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    objects, processed, lookups = load_objects()
    splits, truncation = {}, {}
    for split in ["train_seen", "val_seen", "val_unseen"]:
        splits[split], truncation[split] = load_split(split, objects, processed, lookups)
    if args.train_limit:
        splits["train_seen"] = splits["train_seen"][:args.train_limit]
    if args.val_limit:
        splits["val_seen"] = splits["val_seen"][:args.val_limit]
        splits["val_unseen"] = splits["val_unseen"][:args.val_limit]
    vocab_texts = []
    for sample in splits["train_seen"]:
        vocab_texts.append(replace_names(sample["description"], sample["names"]))
        vocab_texts.extend(replace_names(sample["description"], sample["names"], selected=index)
                           for index in range(len(sample["names"])))
    vocab = Vocabulary(vocab_texts)

    basis_cache = {}
    for split_samples in splits.values():
        for sample in split_samples:
            for landmark_id, landmark, _ in sample["landmarks"]:
                key = (sample["map"], landmark_id)
                if key not in basis_cache:
                    basis_cache[key] = landmark_basis(sample["map"], landmark)
    datasets = {name: HeatmapDataset(samples, vocab, basis_cache) for name, samples in splits.items()}
    loaders = {
        "train_seen": DataLoader(datasets["train_seen"], batch_size=args.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=collate,
                                 generator=torch.Generator().manual_seed(args.seed)),
        "val_seen": DataLoader(datasets["val_seen"], batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=0, collate_fn=collate),
        "val_unseen": DataLoader(datasets["val_unseen"], batch_size=args.batch_size * 2, shuffle=False,
                                 num_workers=0, collate_fn=collate),
    }
    model = RelationalHeatmap(len(vocab.itos), MAX_LANDMARKS,
                              len(ANGLE_NAMES), len(DISTANCE_METERS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(args.epochs):
        model.train(); total_loss = total_ce = total_mass = count = 0
        for raw_batch in loaders["train_seen"]:
            batch = move(raw_batch, device)
            logits, _, _ = forward(model, batch)
            target, within = target_fields(batch, device)
            log_prob = F.log_softmax(logits.flatten(1), dim=1).reshape(-1, SIZE, SIZE)
            probability = log_prob.exp()
            ce = -(target * log_prob).sum(dim=(1, 2)).mean()
            mass_loss = -torch.log((probability * within).sum(dim=(1, 2)).clamp_min(1e-8)).mean()
            loss = ce + 0.5 * mass_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_n = logits.shape[0]; count += batch_n
            total_loss += loss.item() * batch_n; total_ce += ce.item() * batch_n
            total_mass += mass_loss.item() * batch_n
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": total_loss / count, "ce": total_ce / count,
               "mass_loss": total_mass / count, "lr": scheduler.get_last_lr()[0]}
        history.append(row); print(json.dumps(row), flush=True)

    result = {
        "protocol": "PROTOCOL.md", "seed": args.seed, "epochs": args.epochs,
        "device": str(device), "vocab_size": len(vocab.itos),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "samples": {name: len(value) for name, value in splits.items()},
        "truncated_landmark_sets": truncation, "angle_names": ANGLE_NAMES,
        "distance_meters": DISTANCE_METERS,
        "history": history,
    }
    output_rows = {}
    for split in ["val_seen", "val_unseen"]:
        rows = evaluate(model, loaders[split], datasets[split], device)
        output_rows[split] = rows
        result[split] = summarize(rows)
        serializable = [{key: value for key, value in row.items() if key != "heatmap"} for row in rows]
        (args.run_dir / f"{split}_samples.json").write_text(json.dumps(serializable, ensure_ascii=False))
        print(split, json.dumps(result[split]["learned"], ensure_ascii=False), flush=True)
    result["elapsed_seconds"] = round(time.time() - started, 1)
    passed = write_report(args.run_dir / "REPORT.md", result)
    result["passed"] = passed
    (args.run_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                "config": {"size": SIZE, "max_landmarks": MAX_LANDMARKS,
                           "angle_names": ANGLE_NAMES,
                           "distance_meters": DISTANCE_METERS}}, args.run_dir / "model.pt")
    plot_results(args.run_dir, history, result, output_rows["val_unseen"])
    print(json.dumps({"passed": passed, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
