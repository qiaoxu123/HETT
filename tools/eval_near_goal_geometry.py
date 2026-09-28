"""Does the frozen coarse belief already carry usable heading information at 20-40 m?

Diagnostic only: loads a checkpoint, samples independent near-goal validation
states, and compares pure-geometry heading baselines (GT target, coarse Top-1,
Top-K oracle, probability-weighted Top-K) against each other and against the
current fine model on the *same* states. Nothing is trained or modified.

Coordinate convention (matches the verified agent/env convention):
    delta        = target - current          (normalised map coordinates)
    world_bearing = atan2(-delta_y, delta_x)
    relative      = wrap(world_bearing - yaw)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multiagent")
)

MAP_METERS = 410.0
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diagnostics", "near_goal_geometry",
)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rel_heading(coord, cur, yaw):
    """Relative heading from a normalised map coordinate, pure geometry."""
    dx = coord[..., 0] - cur[..., 0]
    dy = coord[..., 1] - cur[..., 1]
    return wrap(np.arctan2(-dy, dx) - yaw)


def angular_error(pred, gt):
    return np.abs(wrap(pred - gt))


def describe(err_deg):
    return dict(
        MAE=float(err_deg.mean()),
        median=float(np.median(err_deg)),
        RMSE=float(np.sqrt(np.mean(err_deg ** 2))),
        w15=float(100.0 * np.mean(err_deg <= 15)),
        w30=float(100.0 * np.mean(err_deg <= 30)),
        w45=float(100.0 * np.mean(err_deg <= 45)),
        w60=float(100.0 * np.mean(err_deg <= 60)),
        w90=float(100.0 * np.mean(err_deg <= 90)),
    )


def collect(args, n_states):
    """Sample independent near-goal states and record model + geometry outputs."""
    from multiagent.parser import parse_args
    from multiagent.main import build_val_dataset
    from multiagent.agent import NavCMTAgent

    sys.argv = [
        "eval_near_goal_geometry", "--mode", "eval", "--world_size", "1", "--seed", "0",
        "--training_stage", "fine", "--feedback", "student",
        "--target_representation", "candidates",
        "--candidate_grid_size", "8", "--candidate_topk", "4",
        "--action_controller", "fine_waypoint", "--use_stop_head",
        "--stop_distance_m", "20", "--stop_threshold", "0.5",
        "--fine_random_start", "--fine_start_min_m", "20",
        "--fine_start_max_m", "40", "--fine_random_yaw",
        "--fine_waypoint_m", "20", "--fine_distance_loss_weight", "1.0",
        "--fine_stop_pos_weight", "1.0", "--fine_one_state_supervision",
        "--fine_stop_positive_fraction", "0.30", "--fine_positive_min_m", "5",
        "--max_episodes", str(n_states), "--batch_size", "8",
        "--max_action_len", "1", "--grid_size", "5",
        "--output_dir", os.path.join(os.path.dirname(OUT_DIR), "_geom_tmp"),
        "--checkpoint", args.checkpoint,
    ]
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    env = build_val_dataset(cfg)["val_unseen"]
    agent = NavCMTAgent(cfg, allow_ngpus=False)
    agent.load(cfg.checkpoint)
    agent.env = env

    et_cls = type(agent.vln_model)
    orig_fwd, orig_obs = et_cls.forward, env._get_obs
    rec = defaultdict(list)
    pending = {}

    def get_obs(poses=None, **kw):
        obs = orig_obs(poses, **kw)
        if poses is None:
            pending["obs"] = obs
        return obs

    def forward(self, **inputs):
        out = orig_fwd(self, **inputs)
        obs = pending.get("obs")
        if obs is None or self.last_topk_coordinates is None:
            return out
        d = inputs["directions"][:, -1]
        yaw = torch.atan2(d[:, 0], d[:, 1]).detach().cpu().numpy()
        topk = self.last_topk_coordinates.detach().cpu().numpy()
        probs = self.last_topk_probs.detach().cpu().numpy()
        fine = (self.last_fine_relative_heading.detach().cpu().numpy()
                if self.last_fine_relative_heading is not None else None)
        for i in range(min(len(obs), topk.shape[0])):
            ob = obs[i]
            rec["eid"].append(ob["id"])
            rec["cur"].append(np.array(ob["position"], dtype=np.float64))
            rec["goal"].append(np.array(ob["normalized_goal"], dtype=np.float64))
            rec["gt_dist"].append(float(ob["distance_to_goal_m"]))
            rec["gt_head"].append(float(ob["direction"]))
            rec["yaw"].append(float(yaw[i]))
            rec["topk"].append(topk[i])
            rec["probs"].append(probs[i])
            if fine is not None:
                rec["fine_head"].append(float(fine[i]))
        return out

    et_cls.forward, env._get_obs = forward, get_obs
    try:
        # next_batch() is a generator function: one generator must be advanced,
        # otherwise every call restarts at ix=0 and re-yields the first batch.
        batches = env.next_batch()
        done = 0
        while done < n_states:
            try:
                next(batches)
            except StopIteration:
                break
            done += env.batch_size
            agent.logs = defaultdict(list)
            agent.loss = 0
            agent.losses = []
            agent.feedback = "student"
            agent.rollout(train_ml=1.0)
    finally:
        et_cls.forward, env._get_obs = orig_fwd, orig_obs

    return {k: np.array(v, dtype=object) if k == "eid" else np.array(v)
            for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--states", type=int, default=1600)
    ap.add_argument("--visualize", type=int, default=30)
    args = ap.parse_args()

    data = collect(args, args.states)
    cur = np.stack(data["cur"]).astype(np.float64)
    goal = np.stack(data["goal"]).astype(np.float64)
    topk = np.stack(data["topk"]).astype(np.float64)
    probs = np.stack(data["probs"]).astype(np.float64)
    gt_h = np.array(data["gt_head"], dtype=np.float64)
    yaw = np.array(data["yaw"], dtype=np.float64)
    gtd = np.array(data["gt_dist"], dtype=np.float64)
    fine_h = np.array(data["fine_head"], dtype=np.float64) if len(data["fine_head"]) else None
    n = gt_h.size
    print(f"\n=== dataset ===\n  N states {n}")
    print(f"  GT range: min {gtd.min():.3f}  mean {gtd.mean():.3f}  max {gtd.max():.3f} m")
    print(f"  yaw: mean {yaw.mean():.4f}  std {yaw.std():.4f}  "
          f"min {yaw.min():.4f}  max {yaw.max():.4f}")

    print("\n=== 3. Baseline A: GT geometry sanity ===")
    gt_geo = rel_heading(goal, cur, yaw)
    a_err = np.degrees(angular_error(gt_geo, gt_h))
    d = describe(a_err)
    print(f"  MAE {d['MAE']:.6f}  median {d['median']:.6f}  RMSE {d['RMSE']:.6f}")
    print(f"  within5 {100*np.mean(a_err<=5):.3f}%  within15 {100*np.mean(a_err<=15):.3f}%")
    corr_a = np.corrcoef(gt_geo, gt_h)[0, 1] if gt_geo.std() > 1e-12 else float("nan")
    print(f"  corr {corr_a:.6f}")

    print("\n=== 4. Baseline B: coarse Top-1 geometry ===")
    top1 = topk[:, 0]
    h1 = rel_heading(top1, cur, yaw)
    e1 = np.degrees(angular_error(h1, gt_h))
    d1 = describe(e1)
    coord1 = np.linalg.norm((top1 - goal) * MAP_METERS, axis=-1)
    print(f"  MAE {d1['MAE']:.3f}  median {d1['median']:.3f}  RMSE {d1['RMSE']:.3f}")
    for k in ("w15", "w30", "w45", "w60", "w90"):
        print(f"  {k} {d1[k]:.2f}%")
    print(f"  corr {np.corrcoef(h1, gt_h)[0,1]:.4f}")
    print(f"  Top1 coordinate error: mean {coord1.mean():.3f} m  median {np.median(coord1):.3f} m")

    print("\n=== 5. Baseline C: Top-4 oracle geometry ===")
    hk = np.stack([rel_heading(topk[:, k], cur, yaw) for k in range(topk.shape[1])], axis=1)
    ek = np.degrees(np.abs(wrap(hk - gt_h[:, None])))
    oracle_e = ek.min(axis=1)
    dc = describe(oracle_e)
    print(f"  MAE {dc['MAE']:.3f}  median {dc['median']:.3f}")
    for k in ("w15", "w30", "w45", "w60", "w90"):
        print(f"  {k} {dc[k]:.2f}%")

    print("\n=== 6. Baseline D: probability-weighted Top-K (circular) ===")
    sb = (probs * np.sin(hk)).sum(1)
    cb = (probs * np.cos(hk)).sum(1)
    hw = np.arctan2(sb, cb)
    ew = np.degrees(angular_error(hw, gt_h))
    dw = describe(ew)
    print(f"  MAE {dw['MAE']:.3f}  corr {np.corrcoef(hw, gt_h)[0,1]:.4f}  within45 {dw['w45']:.2f}%")

    print("\n=== 7. Best hypothesis rank ===")
    best_rank = ek.argmin(axis=1)
    cnt = np.bincount(best_rank, minlength=topk.shape[1])
    for r in range(topk.shape[1]):
        print(f"  rank{r+1}: {100.0*cnt[r]/n:.2f}%")
    print(f"  Top1 IS the best heading hypothesis: {100.0*np.mean(best_rank==0):.2f}%")

    print("\n=== 8. coordinate-error buckets ===")
    print(f"{'bucket':>10} {'n':>6} {'Top1 MAE':>10} {'Top1 w30':>9} {'Top1 w45':>9} "
          f"{'Oracle MAE':>11} {'Orc w30':>8} {'Orc w45':>8}")
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 1e9)):
        m = (coord1 >= lo) & (coord1 < hi)
        if not m.any():
            print(f"{f'{lo}-{hi:.0f}':>10} {0:>6}  (empty)")
            continue
        print(f"{f'{lo}-{hi:.0f}':>10} {int(m.sum()):>6} {e1[m].mean():>10.3f} "
              f"{100*np.mean(e1[m]<=30):>9.2f} {100*np.mean(e1[m]<=45):>9.2f} "
              f"{oracle_e[m].mean():>11.3f} {100*np.mean(oracle_e[m]<=30):>8.2f} "
              f"{100*np.mean(oracle_e[m]<=45):>8.2f}")

    print("\n=== 9. GT-range buckets ===")
    print(f"{'range':>10} {'n':>6} {'Top1 MAE':>10} {'Top1 w30':>9} {'Top1 w45':>9} "
          f"{'Oracle MAE':>11} {'Orc w30':>8} {'Orc w45':>8}")
    for lo, hi in ((20, 25), (25, 30), (30, 35), (35, 40.01)):
        m = (gtd >= lo) & (gtd < hi)
        if not m.any():
            continue
        print(f"{f'{lo}-{hi:.0f}':>10} {int(m.sum()):>6} {e1[m].mean():>10.3f} "
              f"{100*np.mean(e1[m]<=30):>9.2f} {100*np.mean(e1[m]<=45):>9.2f} "
              f"{oracle_e[m].mean():>11.3f} {100*np.mean(oracle_e[m]<=30):>8.2f} "
              f"{100*np.mean(oracle_e[m]<=45):>8.2f}")

    print("\n=== 10. Top-K direction diversity ===")
    K = topk.shape[1]
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    sep = np.stack([np.degrees(np.abs(wrap(hk[:, a] - hk[:, b]))) for a, b in pairs], axis=1)
    mean_sep = sep.mean(axis=1)
    max_sep = sep.max(axis=1)
    print(f"  mean pairwise heading separation: mean {mean_sep.mean():.2f}  "
          f"median {np.median(mean_sep):.2f}  P25 {np.percentile(mean_sep,25):.2f}  "
          f"P75 {np.percentile(mean_sep,75):.2f} deg")
    for t in (15, 30, 45, 90):
        print(f"  % states with max separation > {t:>2} deg: {100.0*np.mean(max_sep>t):.2f}%")

    print("\n=== 11. confidence vs heading correctness ===")
    p1 = probs[:, 0]
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    sp = np.corrcoef(np.argsort(np.argsort(p1)), np.argsort(np.argsort(e1)))[0, 1]
    print(f"  Spearman(top1 prob, heading error) = {sp:+.4f}")
    print(f"  Spearman(-entropy,     heading error) = "
          f"{np.corrcoef(np.argsort(np.argsort(-ent)), np.argsort(np.argsort(e1)))[0,1]:+.4f}")
    for lo, hi in ((0, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 1.01)):
        m = (p1 >= lo) & (p1 < hi)
        if not m.any():
            continue
        print(f"  conf {lo:.2f}-{hi:.2f}: n={int(m.sum()):>5}  heading MAE {e1[m].mean():7.3f}  "
              f"within45 {100*np.mean(e1[m]<=45):5.2f}%")

    if fine_h is not None:
        print("\n=== 12. same-state comparison ===")
        ef = np.degrees(angular_error(fine_h, gt_h))
        df = describe(ef)
        print(f"{'baseline':>18} {'MAE':>9} {'median':>8} {'corr':>8} "
              f"{'w15':>7} {'w30':>7} {'w45':>7} {'w90':>7}")
        rows = [("GT geometry", a_err, gt_geo), ("Top1 geometry", e1, h1),
                ("Top4 oracle", oracle_e, None), ("Weighted TopK", ew, hw),
                ("Fine model", ef, fine_h)]
        for name, e, vec in rows:
            c = f"{np.corrcoef(vec, gt_h)[0,1]:+.4f}" if vec is not None else "   n/a"
            print(f"{name:>18} {e.mean():>9.3f} {np.median(e):>8.3f} {c:>8} "
                  f"{100*np.mean(e<=15):>7.2f} {100*np.mean(e<=30):>7.2f} "
                  f"{100*np.mean(e<=45):>7.2f} {100*np.mean(e<=90):>7.2f}")

    print("\n=== 14. coordinate-oracle vs heading-oracle candidate ===")
    ci = np.stack([np.linalg.norm((topk[:, k] - goal) * MAP_METERS, axis=-1)
                   for k in range(K)], axis=1).argmin(axis=1)
    hi_ = ek.argmin(axis=1)
    coord_oracle_err = ek[np.arange(n), ci]
    head_oracle_err = oracle_e
    print(f"  coordinate-oracle heading MAE : {coord_oracle_err.mean():.3f} deg")
    print(f"  heading-oracle    heading MAE : {head_oracle_err.mean():.3f} deg")
    print(f"  within30 {100*np.mean(coord_oracle_err<=30):.2f}%   "
          f"within45 {100*np.mean(coord_oracle_err<=45):.2f}%")
    print(f"  same candidate ratio           : {100.0*np.mean(ci==hi_):.2f}%")
    print(f"  coordinate-oracle candidate error: mean "
          f"{np.stack([np.linalg.norm((topk[:,k]-goal)*MAP_METERS,axis=-1) for k in range(K)],axis=1)[np.arange(n),ci].mean():.3f} m")

    if args.visualize > 0:
        try:
            visualize(cur, goal, topk, probs, yaw, gt_h, gtd, coord1, e1, oracle_e,
                      args.visualize)
        except Exception as exc:  # pragma: no cover - plotting is best effort
            print(f"  (visualization skipped: {exc})")


def visualize(cur, goal, topk, probs, yaw, gt_h, gtd, coord1, e1, oracle_e, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    good = np.where(e1 <= 15)[0]
    mid = np.where((e1 > 45) & (oracle_e <= 30))[0]
    bad = np.where(oracle_e > 60)[0]
    rng = np.random.default_rng(0)
    picks = []
    for name, pool in (("good", good), ("topk_fixable", mid), ("unusable", bad)):
        take = rng.choice(pool, size=min(10, pool.size), replace=False) if pool.size else []
        picks.extend((name, int(i)) for i in take)
    picks = picks[:count]
    for name, i in picks:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        ax.scatter(*goal[i], c="green", marker="*", s=120, label="GT target")
        ax.scatter(*cur[i], c="black", marker="o", s=40, label="UAV")
        for k in range(topk.shape[1]):
            ax.scatter(*topk[i, k], c="C1", s=25)
            ax.annotate(f"{k+1}", topk[i, k], fontsize=7)
        ax.arrow(cur[i][0], cur[i][1], 0.03 * np.cos(yaw[i]), -0.03 * np.sin(yaw[i]),
                 head_width=0.006, color="black")
        ax.arrow(cur[i][0], cur[i][1], 0.06 * np.cos(gt_h[i] + yaw[i]),
                 -0.06 * np.sin(gt_h[i] + yaw[i]), head_width=0.006, color="green")
        ax.arrow(cur[i][0], cur[i][1], 0.06 * np.cos(topk_h := (np.arctan2(
            -(topk[i, 0, 1] - cur[i][1]), topk[i, 0, 0] - cur[i][0]))),
                 -0.06 * np.sin(topk_h), head_width=0.006, color="C1")
        ax.set_title(f"{name}  range {gtd[i]:.1f} m\n"
                     f"coord err {coord1[i]:.1f} m | top1 hdg err {e1[i]:.1f}° | "
                     f"oracle {oracle_e[i]:.1f}°\n"
                     f"probs {np.round(probs[i], 3).tolist()}", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}_{i:05d}.png"), dpi=90)
        plt.close(fig)
    print(f"\n  saved {len(picks)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
