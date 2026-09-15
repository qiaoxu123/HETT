"""CPU-only RGB boundary evidence probe; NOT semantic recognition or navigation.

Fixed map-round-robin 10 distinct validation objects with named landmarks, 3 recorded
trajectory frames each. No selection on visibility or results. Target coordinates
are never used. RGB edges are computed before accessing landmark contours.
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.maps.landmark_map import LandmarkMap
from multiagent.observation import cropclient
from multiagent.defaultpaths import ORTHO_IMAGE_DIR

OUT = ROOT / 'runs/contour_probe_balanced'
OUT.mkdir(parents=True, exist_ok=True)
N = 224
SHIFTS = [(0, 0), (16, 0), (-16, 0), (0, 16), (0, -16),
          (16, 16), (-16, 16), (16, -16), (-16, -16)]

def valid_pixels(rgb):
    black = (np.max(rgb,axis=2)==0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(black,8)
    border = np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    invalid = np.zeros(black.shape,np.uint8)
    for label in border:
        if label and stats[label,cv2.CC_STAT_AREA]>=100:
            invalid[labels==label] = 1
    return 1-cv2.dilate(invalid,np.ones((9,9),np.uint8))

def edge_distance(rgb, low=50, high=150):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), low, high)
    valid = valid_pixels(rgb)
    edges[valid == 0] = 0  # exclude black raster no-data boundaries
    return edges, cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

def boundary_points(contour):
    # Sample original polygon segments, never artificial crop-frame boundaries.
    pieces = []
    for a, b in zip(contour, np.roll(contour, -1, axis=0)):
        count = max(2, int(np.ceil(np.linalg.norm(b-a))))
        pieces.append(a + np.arange(count)[:, None] / count * (b-a))
    points = np.concatenate(pieces)
    # Common support ensures every shifted control uses exactly the same points.
    keep = ((points >= 18) & (points < N-18)).all(axis=1)
    return points[keep], float(keep.mean())

def scores(distance, points):
    result = []
    for dx, dy in SHIFTS:
        p = np.rint(points + [dx, dy]).astype(int)
        result.append(float(np.mean(distance[p[:, 1], p[:, 0]] <= 2)))
    return result

def main():
    objects = get_city_refer_objects()
    chosen, seen, buckets = [], set(), {}
    for t in load_mturk_trajectories('val_unseen', 'all', 50):
        key = (t.map_name, t.object_id)
        names = objects[t.map_name][t.object_id].processed_descriptions[t.desc_id].landmarks
        if key not in seen and names:
            buckets.setdefault(t.map_name, []).append((t, names))
            seen.add(key)
    while len(chosen) < 10:
        added = False
        for map_name in sorted(buckets):
            if buckets[map_name] and len(chosen) < 10:
                chosen.append(buckets[map_name].pop(0))
                added = True
        if not added: break
    rows, displays = [], []
    cropclient._raster_cache, cropclient._rgb_cache = {}, {}
    for ti, (t, names) in enumerate(chosen):
        if t.map_name not in cropclient._raster_cache:
            cropclient._raster_cache[t.map_name] = rasterio.open(ORTHO_IMAGE_DIR / (t.map_name+'.tif'))
            cropclient._rgb_cache[t.map_name] = cv2.cvtColor(cv2.imread(str(ORTHO_IMAGE_DIR / (t.map_name+'.png'))), cv2.COLOR_BGR2RGB)
        landmarks = LandmarkMap._search_landmarks_by_name(t.map_name, names)
        for phase, idx in zip(['start', 'middle', 'end'], [0, len(t.trajectory)//2, len(t.trajectory)-1]):
            pose = t.trajectory[idx].xyzyaw
            rgb = cropclient.crop_image(t.map_name, pose, (N, N), 'rgb')
            edges, distance = edge_distance(rgb)
            valid = valid_pixels(rgb)
            corners = cropclient._compute_view_area_corners_rowcol(t.map_name, pose)[:, ::-1].copy()
            transform = cv2.getPerspectiveTransform(corners, np.float32([[0,0],[N-1,0],[N-1,N-1],[0,N-1]]))
            candidates, outlines = [], []
            for lm in landmarks:
                rc = np.float32([cropclient._raster_cache[t.map_name].index(x,y)[::-1] for x,y in lm.contour])
                contour = cv2.perspectiveTransform(rc[None], transform)[0]
                points, fraction = boundary_points(contour)
                if len(points):
                    keep = np.ones(len(points),dtype=bool)
                    for dx,dy in SHIFTS:
                        p = np.rint(points+[dx,dy]).astype(int)
                        keep &= valid[p[:,1],p[:,0]].astype(bool)
                        # Same valid support for the flipped-image negative control.
                        keep &= valid[p[:,1],N-1-p[:,0]].astype(bool)
                    points = points[keep]
                outlines.append(contour)
                item = {'name': lm.name, 'boundary_points':len(points), 'interior_boundary_fraction':fraction}
                if len(points) >= 20:
                    for tag, lo, hi in [('default',50,150),('low',30,90),('high',80,200)]:
                        dist = distance if tag == 'default' else edge_distance(rgb, lo, hi)[1]
                        item[tag] = scores(dist, points)
                    # Wrong-image control: same boundary, horizontally flipped RGB.
                    item['flipped_rgb'] = scores(edge_distance(rgb[:, ::-1].copy())[1], points)
                candidates.append(item)
            row = {'frame':len(rows), 'trajectory_group':ti, 'map':t.map_name, 'object_id':t.object_id,
                   'phase':phase, 'trajectory_index':idx, 'pose':list(pose), 'landmarks':candidates}
            rows.append(row)
            displays.append((rgb, edges, outlines))
            cv2.imwrite(str(OUT / f'frame_{len(rows)-1:02d}.png'), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            print(f'frame {len(rows)}/30', flush=True)
    summary = {'frames':len(rows), 'trajectories':len(chosen), 'maps':sorted({t.map_name for t,_ in chosen}),
               'protocol':'10 distinct val_unseen objects with named landmarks, sorted-map round-robin, original within-map order; start/middle/end recorded human trajectory at 50m. Not model rollouts. Boundary evidence only; no target coordinates, semantic masks, depth or training.',
               'shift_pixels':16, 'edge_tolerance_pixels':2, 'minimum_boundary_points':20}
    for tag in ['default','low','high','flipped_rgb']:
        pairs = [(r,c) for r in rows for c in r['landmarks'] if tag in c]
        values = np.array([c[tag] for r,c in pairs])
        summary[tag] = {'eligible_frame_landmarks':len(values),
            'true_boundary_support':float(values[:,0].mean()),
            'shifted_boundary_support':float(values[:,1:].mean()),
            'strict_top1_fraction':float((values[:,0] > values[:,1:].max(axis=1)).mean()),
            'true_beats_control_mean_fraction':float((values[:,0] > values[:,1:].mean(axis=1)).mean())}
    eligible = [r for r in rows if any('default' in c for c in r['landmarks'])]
    summary['frames_with_sufficient_interior_boundary'] = len(eligible)
    # Descriptive sampling uncertainty, clustered by trajectory (not independent frames).
    group_deltas = []
    for g in range(len(chosen)):
        ds = [c['default'][0]-np.mean(c['default'][1:]) for r in rows if r['trajectory_group']==g for c in r['landmarks'] if 'default' in c]
        if ds: group_deltas.append(float(np.mean(ds)))
    rng = np.random.default_rng(0)
    boot = np.mean(rng.choice(group_deltas, (5000,len(group_deltas)), replace=True),axis=1)
    summary['trajectory_mean_support_gain'] = float(np.mean(group_deltas))
    summary['trajectory_bootstrap_95_interval'] = np.quantile(boot,[.025,.975]).tolist()
    (OUT/'results.json').write_text(json.dumps({'summary':summary,'frames':rows},indent=2))
    fig, axes = plt.subplots(6,5,figsize=(15,18))
    for ax,r,(rgb,edges,contours) in zip(axes.flat,rows,displays):
        ax.imshow(rgb)
        for contour in contours: ax.plot(*np.vstack([contour,contour[0]]).T,color='lime',lw=.8)
        ax.set(xlim=(0,N),ylim=(N,0),title=f"#{r['frame']} {r['phase']}")
        ax.axis('off')
    fig.suptitle('ALL 30 frames: green = projected supplied landmark boundary (NOT extracted)',fontsize=14)
    fig.tight_layout(); fig.savefig(OUT/'all_frames.png',dpi=120); plt.close(fig)
    ranked = sorted(eligible,key=lambda r:np.mean([c['default'][0]-np.mean(c['default'][1:]) for c in r['landmarks'] if 'default' in c]))
    selected = [ranked[-1],ranked[len(ranked)//2],ranked[0]]
    fig, axes = plt.subplots(3,3,figsize=(11,11))
    for axs,r,label in zip(axes,selected,['Largest gain','Median gain','Smallest gain']):
        rgb,edges,contours=displays[r['frame']]
        axs[0].imshow(rgb); axs[1].imshow(edges,cmap='gray'); axs[2].imshow(rgb)
        for contour in contours:
            line=np.vstack([contour,contour[0]])
            axs[2].plot(*line.T,color='lime',lw=1)
            axs[2].plot(*(line+[16,0]).T,color='magenta',lw=.8,ls='--')
        axs[0].set_title(f"{label}: frame {r['frame']}")
        axs[1].set_title('RGB-only Canny edges')
        axs[2].set_title('Green: map boundary; pink: shifted')
        for ax in axs: ax.set(xlim=(0,N),ylim=(N,0)); ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT/'examples.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(3); tags=['default','low','high']
    ax.bar(x-.18,[summary[t]['true_boundary_support']*100 for t in tags],.36,label='Map boundary')
    ax.bar(x+.18,[summary[t]['shifted_boundary_support']*100 for t in tags],.36,label='Shifted controls')
    ax.set_xticks(x); ax.set_xticklabels(['Default edges','Lower threshold','Higher threshold'])
    ax.set_ylabel('Boundary within 2 px of RGB edge (%)'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=150); plt.close(fig)
    for raster in cropclient._raster_cache.values(): raster.close()
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
