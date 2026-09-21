"""Train explicit selection over landmark direction-distance candidates."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiagent.mapdata import MAP_BOUNDS  # noqa: E402
from multiagent.models.relational_heatmap import HETTBertCandidateSelector  # noqa: E402
from scripts.train_relational_heatmap import (  # noqa: E402
    ANGLE_NAMES, DISTANCE_METERS, MAX_LANDMARKS, MAX_TOKENS, NMS_METERS, SIZE,
    landmark_basis, load_objects, load_split, pair_field, replace_names,
)

CANDIDATES_PER_LANDMARK = len(ANGLE_NAMES) * len(DISTANCE_METERS)
TOTAL_CANDIDATES = MAX_LANDMARKS * CANDIDATES_PER_LANDMARK + 1
DEFAULT_CHECKPOINT = Path("/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911/checkpoints/best_val_unseen")
CHECKPOINT_SHA256 = "657e6bde266b6e3c678370a9e52cc8366662490e411ea6c2a2549a4488b2ea56"


def bert_text(text):
    return (text.replace("<ref>", "the referenced landmark")
                .replace("<other>", "another named landmark")
                .replace("<landmark>", "a named landmark"))


def attach_texts(splits):
    all_texts = set()
    for samples in splits.values():
        for sample in samples:
            names = sample["names"]
            sample["global_text"] = bert_text(replace_names(sample["description"], names))
            sample["ref_texts"] = [bert_text(replace_names(sample["description"], names, selected=index))
                                   for index in range(len(names))]
            all_texts.add(sample["global_text"]); all_texts.update(sample["ref_texts"])
    return sorted(all_texts)


def load_or_create_embeddings(texts, cache_path, checkpoint, device):
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (cache.get("checkpoint_sha256") != CHECKPOINT_SHA256 or cache["texts"] != texts
                or cache.get("pooling") != "masked_mean_last_hidden"):
            raise RuntimeError("embedding cache does not match checkpoint or text inventory")
        return cache["embeddings"]
    from transformers import AutoModel, AutoTokenizer
    print(f"encoding {len(texts)} unique marked instructions with HETT BERT", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("bert."): value for key, value in saved["lang_model"]["state_dict"].items()
             if key.startswith("bert.")}
    bert.load_state_dict(state, strict=True); del saved, state
    bert = bert.to(device).eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 128):
            encoded = tokenizer(texts[start:start + 128], padding=True, truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt").to(device)
            sequence = bert(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            chunks.append(pooled.half().cpu())
            if start % 4096 == 0: print(f"encoded {start}/{len(texts)}", flush=True)
    embeddings = torch.cat(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"texts": texts, "embeddings": embeddings, "checkpoint": str(checkpoint),
                "pooling": "masked_mean_last_hidden",
                "checkpoint_sha256": CHECKPOINT_SHA256}, cache_path)
    return embeddings


def basis_peaks(basis):
    angles, distances = basis
    fields = angles[:, None].astype(np.float32) * distances[None].astype(np.float32)
    flat = fields.reshape(CANDIDATES_PER_LANDMARK, -1).argmax(axis=1)
    return np.stack(np.unravel_index(flat, (SIZE, SIZE)), axis=-1).astype(np.float32)


class CandidateDataset(Dataset):
    def __init__(self, samples, embeddings, text_to_index, candidate_cache):
        self.samples = samples
        self.embeddings = embeddings
        self.text_to_index = text_to_index
        self.candidate_cache = candidate_cache
        self.pair_cache = {}
        for sample in self.samples:
            sample["global_embedding_index"] = text_to_index[sample["global_text"]]
            sample["ref_embedding_indices"] = [text_to_index[text] for text in sample["ref_texts"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        candidates = np.zeros((TOTAL_CANDIDATES, 2), dtype=np.float32)
        candidate_valid = np.zeros(TOTAL_CANDIDATES, dtype=np.float32)
        ref_embeddings = np.zeros((MAX_LANDMARKS, 768), dtype=np.float16)
        landmark_valid = np.zeros(MAX_LANDMARKS, dtype=np.float32)
        for slot, (landmark_id, _, _) in enumerate(sample["landmarks"]):
            start = slot * CANDIDATES_PER_LANDMARK
            candidates[start:start + CANDIDATES_PER_LANDMARK] = self.candidate_cache[(sample["map"], landmark_id)]
            candidate_valid[start:start + CANDIDATES_PER_LANDMARK] = 1
            ref_embeddings[slot] = self.embeddings[sample["ref_embedding_indices"][slot]].numpy()
            landmark_valid[slot] = 1
        pair_valid = float(len(sample["landmarks"]) >= 2)
        if pair_valid:
            key = (sample["map"], tuple(item[0] for item in sample["landmarks"]))
            if key not in self.pair_cache:
                field = pair_field(sample["map"], sample["landmarks"])[0]
                flat = int(field.argmax())
                self.pair_cache[key] = np.asarray(divmod(flat, SIZE), dtype=np.float32)
            candidates[-1] = self.pair_cache[key]
            candidate_valid[-1] = 1
        bounds = MAP_BOUNDS[sample["map"]]
        target_col = (sample["target"][0] - bounds.x_min) / (bounds.x_max - bounds.x_min) * (SIZE - 1)
        target_row = (bounds.y_max - sample["target"][1]) / (bounds.y_max - bounds.y_min) * (SIZE - 1)
        return {
            "candidates": candidates, "candidate_valid": candidate_valid,
            "ref_embeddings": ref_embeddings,
            "global_embedding": self.embeddings[sample["global_embedding_index"]].numpy(),
            "landmark_valid": landmark_valid, "pair_valid": np.asarray(pair_valid, dtype=np.float32),
            "target_rc": np.asarray([target_row, target_col], dtype=np.float32),
            "cell_xy": np.asarray([(bounds.x_max - bounds.x_min) / (SIZE - 1),
                                   (bounds.y_max - bounds.y_min) / (SIZE - 1)], dtype=np.float32),
            "index": index,
        }


def collate(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = values if key == "index" else torch.from_numpy(np.stack(values))
    return result


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def candidate_distances(batch):
    delta = batch["candidates"] - batch["target_rc"].unsqueeze(1)
    dy = delta[..., 0] * batch["cell_xy"][:, None, 1]
    dx = delta[..., 1] * batch["cell_xy"][:, None, 0]
    distance = torch.sqrt(dx.square() + dy.square())
    return distance.masked_fill(~batch["candidate_valid"].bool(), float("inf"))


def selector_loss(logits, batch):
    distances = candidate_distances(batch)
    positive = distances <= 20.0
    no_positive = ~positive.any(dim=1)
    if no_positive.any():
        nearest = distances.argmin(dim=1)
        positive[no_positive] = F.one_hot(nearest[no_positive], logits.shape[1]).bool()
    log_prob = F.log_softmax(logits, dim=1)
    positive_log_mass = torch.logsumexp(log_prob.masked_fill(~positive, -1e4), dim=1)
    return -positive_log_mass.mean()


def select_points(logits, candidates, valid, cell_xy, count=5):
    order = np.argsort(-logits)
    selected = []
    for candidate_index in order:
        if not valid[candidate_index]:
            continue
        point = candidates[candidate_index]
        if all(math.hypot((point[0] - prior[0]) * cell_xy[1],
                          (point[1] - prior[1]) * cell_xy[0]) > NMS_METERS
               for prior in selected):
            selected.append(point)
            if len(selected) == count:
                break
    return selected


def point_error(point, target, cell_xy):
    return math.hypot((point[0] - target[0]) * cell_xy[1],
                      (point[1] - target[1]) * cell_xy[0])


def evaluate(model, loader, dataset, device):
    model.eval(); rows = []
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            logits = model(batch["ref_embeddings"], batch["global_embedding"],
                           batch["landmark_valid"], batch["pair_valid"])
            probabilities = logits.softmax(dim=1)
            distances = candidate_distances(batch)
            mass20 = (probabilities * (distances <= 20)).sum(dim=1).cpu().numpy()
            logits = logits.cpu().numpy(); candidates = raw["candidates"].numpy()
            valid = raw["candidate_valid"].numpy(); targets = raw["target_rc"].numpy()
            cells = raw["cell_xy"].numpy(); oracle = distances.min(dim=1).values.cpu().numpy()
            for offset, sample_index in enumerate(raw["index"]):
                sample = dataset.samples[sample_index]
                points = select_points(logits[offset], candidates[offset], valid[offset], cells[offset])
                errors = [point_error(point, targets[offset], cells[offset]) for point in points] or [float("inf")]
                centroid_errors = []
                bounds = MAP_BOUNDS[sample["map"]]
                for _, landmark, _ in sample["landmarks"]:
                    x, y = landmark["position"][:2]
                    point = np.asarray([(bounds.y_max - y) / (bounds.y_max - bounds.y_min) * (SIZE - 1),
                                        (x - bounds.x_min) / (bounds.x_max - bounds.x_min) * (SIZE - 1)])
                    centroid_errors.append(point_error(point, targets[offset], cells[offset]))
                centroid_errors = centroid_errors or [float("inf")]
                rows.append({
                    "index": sample_index, "map": sample["map"], "description": sample["description"],
                    "relation": sample["relation"], "landmark_count": len(sample["landmarks"]),
                    "learned_top1_m": errors[0], "learned_top5_m": min(errors),
                    "centroid_top1_m": centroid_errors[0], "centroid_top5_m": min(centroid_errors[:5]),
                    "oracle_top1_m": float(oracle[offset]), "oracle_top5_m": float(oracle[offset]),
                    "mass20": float(mass20[offset]), "target_rc": targets[offset].tolist(),
                    "cell_xy": cells[offset].tolist(), "points_rc": [point.tolist() for point in points],
                })
    return rows


def metric(rows, prefix):
    top1 = np.asarray([row[f"{prefix}_top1_m"] for row in rows])
    top5 = np.asarray([row[f"{prefix}_top5_m"] for row in rows])
    finite = top1[np.isfinite(top1)]
    result = {"n": len(rows), "top1_hit20": round(100 * float(np.mean(top1 <= 20)), 2),
              "top5_recall20": round(100 * float(np.mean(top5 <= 20)), 2),
              "median_top1_m": round(float(np.median(finite)), 2) if len(finite) else None,
              "p90_top1_m": round(float(np.percentile(finite, 90)), 2) if len(finite) else None}
    if prefix == "learned": result["mean_mass20"] = round(float(np.mean([row["mass20"] for row in rows])), 4)
    return result


def summarize(rows):
    result = {key: metric(rows, key) for key in ["centroid", "oracle", "learned"]}
    result["by_relation"] = {}
    for relation in sorted({row["relation"] for row in rows}):
        result["by_relation"][relation] = metric([row for row in rows if row["relation"] == relation], "learned")
    result["by_landmark_count"] = {
        "one": metric([row for row in rows if row["landmark_count"] == 1], "learned"),
        "two_plus": metric([row for row in rows if row["landmark_count"] >= 2], "learned"),
    }
    return result


def write_report(path, result):
    lines = ["# 显式几何候选选择器", ""]
    for split in ["val_seen", "val_unseen"]:
        lines += [f"## {split}", "", "| 方法 | Top-1 | Top-5 | 中位误差 |", "|---|---:|---:|---:|"]
        for key, label in [("centroid", "地标中心"), ("learned", "候选选择器"), ("oracle", "候选上限（特权）")]:
            row = result[split][key]
            lines.append(f"| {label} | {row['top1_hit20']:.2f}% | {row['top5_recall20']:.2f}% | {row['median_top1_m']} m |")
        lines += ["", "| unseen关系" if split == "val_unseen" else "| 关系", ""] if False else []
    unseen = result["val_unseen"]["learned"]; seen = result["val_seen"]["learned"]
    gap = seen["top1_hit20"] - unseen["top1_hit20"]
    passed = unseen["top1_hit20"] >= 35 and unseen["top5_recall20"] >= 70 and gap <= 15
    lines += ["## 判定", "", f"unseen Top-1 {unseen['top1_hit20']:.2f}%，Top-5 {unseen['top5_recall20']:.2f}%，seen-unseen差 {gap:+.2f} 点。", "",
              "**通过。**" if passed else "**未通过。**", "", "候选上限使用真值选最近候选，仅用于诊断。"]
    path.write_text("\n".join(lines) + "\n"); return passed


def plot_metrics(path, result):
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["centroid", "learned", "oracle"]; x = np.arange(3); width = .35
    for offset, split in enumerate(["val_seen", "val_unseen"]):
        values = [result[split][key]["top1_hit20"] for key in methods]
        ax.bar(x + (offset - .5) * width, values, width, label=split)
    ax.set_xticks(x, methods); ax.set_ylabel("Top-1 Hit@20 (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2701); parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--hett-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); started = time.time()
    objects, processed, lookups = load_objects(); splits = {}
    for split in ["train_seen", "val_seen", "val_unseen"]:
        splits[split], _ = load_split(split, objects, processed, lookups)
    if args.train_limit: splits["train_seen"] = splits["train_seen"][:args.train_limit]
    if args.val_limit:
        splits["val_seen"] = splits["val_seen"][:args.val_limit]; splits["val_unseen"] = splits["val_unseen"][:args.val_limit]
    texts = attach_texts(splits)
    embeddings = load_or_create_embeddings(texts, args.embedding_cache, args.hett_checkpoint, device)
    text_to_index = {text: index for index, text in enumerate(texts)}
    cache = {}
    for samples in splits.values():
        for sample in samples:
            for landmark_id, landmark, _ in sample["landmarks"]:
                key = (sample["map"], landmark_id)
                if key not in cache: cache[key] = basis_peaks(landmark_basis(sample["map"], landmark))
    datasets = {key: CandidateDataset(value, embeddings, text_to_index, cache) for key, value in splits.items()}
    loaders = {
        "train_seen": DataLoader(datasets["train_seen"], batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                                 generator=torch.Generator().manual_seed(args.seed)),
        "val_seen": DataLoader(datasets["val_seen"], batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate),
        "val_unseen": DataLoader(datasets["val_unseen"], batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate),
    }
    model = HETTBertCandidateSelector(MAX_LANDMARKS, CANDIDATES_PER_LANDMARK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs); history = []
    for epoch in range(args.epochs):
        model.train(); total = count = 0
        for raw in loaders["train_seen"]:
            batch = move(raw, device); logits = model(batch["ref_embeddings"], batch["global_embedding"],
                                                      batch["landmark_valid"], batch["pair_valid"])
            loss = selector_loss(logits, batch); optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            total += loss.item() * logits.shape[0]; count += logits.shape[0]
        scheduler.step(); row = {"epoch": epoch + 1, "loss": total / count, "lr": scheduler.get_last_lr()[0]}
        history.append(row); print(json.dumps(row), flush=True)
    result = {"protocol": "PROTOCOL.md", "seed": args.seed, "epochs": args.epochs,
              "hett_checkpoint": str(args.hett_checkpoint), "hett_checkpoint_sha256": CHECKPOINT_SHA256,
              "pooling": "masked_mean_last_hidden",
              "parameters": sum(p.numel() for p in model.parameters()), "samples": {k: len(v) for k, v in splits.items()},
              "history": history}
    for split in ["val_seen", "val_unseen"]:
        rows = evaluate(model, loaders[split], datasets[split], device); result[split] = summarize(rows)
        (args.run_dir / f"{split}_samples.json").write_text(json.dumps(rows, ensure_ascii=False))
        print(split, json.dumps(result[split]["learned"]), flush=True)
    result["elapsed_seconds"] = round(time.time() - started, 1); result["passed"] = write_report(args.run_dir / "REPORT.md", result)
    (args.run_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    torch.save({"model": model.state_dict(), "hett_checkpoint_sha256": CHECKPOINT_SHA256},
               args.run_dir / "model.pt")
    plot_metrics(args.run_dir / "metrics.png", result)
    print(json.dumps({"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__": main()
