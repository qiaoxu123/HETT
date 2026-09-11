"""Predeclared local RGB vs geometry probe; never changes the navigation policy.

Collection uses student rollout, not teacher/target-centred crops. The learned
probe sees no target metadata, landmark inventory, map identity or world coords.
Train labels live separately from features. Evaluation uses unseen-map states.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'multiagent')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OLD = Path('/home/tenant2/Workspace/htnav-repro')
SIDE = 49.0
GRID = 7
TRAIN_STEPS = [12, 15, 18]
SEEDS = [17, 29, 43]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_args(checkpoint):
    from multiagent.parser import parse_args
    saved = sys.argv
    sys.argv = ['probe', '--mode', 'eval', '--batch_size', '2',
                '--disable_task_interaction', '--checkpoint', str(checkpoint)]
    args = parse_args()
    sys.argv = saved
    return args


def select_tasks(tasks, n):
    # Equal-map round-robin, randomized within map, independent of targets.
    groups = defaultdict(list)
    for item in tasks:
        groups[item.map_name].append(item)
    rng = random.Random(20260911)
    for items in groups.values():
        rng.shuffle(items)
    selected = []
    while len(selected) < n and any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                selected.append(groups[name].pop())
                if len(selected) == n:
                    break
    return selected


class Cropper:
    def __init__(self):
        self.meta = json.loads((OLD / 'data/rgbd_npy/meta.json').read_text())
        self.arrays = {}

    def get(self, name, pose):
        import cv2
        from multiagent.mapdata import GROUND_LEVEL
        if name not in self.arrays:
            self.arrays[name] = [np.load(OLD / folder / f'{name}_{suffix}.npy', mmap_mode='r')
                                 for folder, suffix in [('data/rgbd_npy', 'rgb'),
                                                        ('data/rgbd_npy', 'h'),
                                                        ('data/cloudgrid', 'density')]]
        rgb, height, density = self.arrays[name]
        a, b, c, d, e, f = self.meta[name]['transform']
        assert b == 0 and d == 0
        # This north-up square fits within the current downward-camera footprint
        # for every yaw. Thus no unseen surrounding/global point cloud is exposed.
        assert SIDE / np.sqrt(2) <= pose.z - GROUND_LEVEL[name]
        axis = (np.arange(224) + .5) / 224 - .5
        xs = pose.x + axis * SIDE
        ys = pose.y - axis * SIDE
        cols, rows = np.meshgrid((xs-c)/a, (ys-f)/e)
        # Bound the source view so OpenCV never converts whole-city memmaps.
        x0, x1 = max(0, int(cols.min())-2), min(rgb.shape[1], int(cols.max())+3)
        y0, y1 = max(0, int(rows.min())-2), min(rgb.shape[0], int(rows.max())+3)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((224,224,3), np.uint8), np.zeros((49,5), np.float32)
        mx, my = (cols-x0).astype('f4'), (rows-y0).astype('f4')
        def remap(arr):
            return cv2.remap(np.asarray(arr[y0:y1,x0:x1]), mx, my, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        image, h, den = remap(rgb), remap(height), remap(density)
        valid = (den > 0) & (h > -9998)
        h = np.where(valid, np.clip(h-GROUND_LEVEL[name], -10, 80), 0)
        den = np.log1p(np.maximum(den, 0))
        def patches(arr):
            return arr.reshape(7,32,7,32).transpose(0,2,1,3).reshape(49,-1)
        hp, dp, vp = patches(h), patches(den), patches(valid)
        geometry = np.stack([hp.mean(1), hp.std(1), hp.max(1), dp.mean(1), vp.mean(1)],1)
        return image, geometry.astype('f4')


@torch.no_grad()
def collect(out, checkpoint, ntrain, nval):
    import multiagent.env as env_module
    from multiagent.agent import NavCMTAgent
    torch.set_num_threads(2)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    args = make_args(checkpoint)
    policy = NavCMTAgent(args, rank=-1, allow_ngpus=False)
    policy.load(str(checkpoint))
    for model in [policy.vln_model, policy.lang_model, policy.vision_model]:
        model.eval()
    cropper = Cropper()
    original_loader = env_module.load_mturk_trajectories
    seen_maps = set()
    for split, count in [('train_seen', ntrain), ('val_unseen', nval)]:
        if (out / f'{split}_features.npz').exists():
            seen_maps.update(json.loads((out / f'{split}_manifest.json').read_text())['maps'])
            continue
        tasks = select_tasks(original_loader(split, 'all', args.altitude), count)
        env_module.load_mturk_trajectories = lambda *a, **k: tasks
        env = env_module.CityNavBatch(split, args, batch_size=2, seed=0)
        env_module.load_mturk_trajectories = original_loader
        maps = sorted({ep.map_name for ep in env.data})
        if split == 'val_unseen':
            assert not seen_maps.intersection(maps), 'map split leakage'
        else:
            seen_maps.update(maps)
        policy.env = env
        policy.feedback = 'student'
        policy.env_name = 'test_probe'  # disable all GT-dependent loss computation
        all_features, all_geo, all_lang = [], [], []
        labels, metadata, predictions = [], [], {}
        start = time.time()
        for batch_num, _ in enumerate(env):
            policy.loss = 0
            trajectories = policy.rollout()
            for ep, traj in zip(env.batch, trajectories):
                key = str(ep.id)
                if key in predictions:
                    continue
                # Persist actual trajectories independently of the local predictor.
                predictions[key] = traj
                tok = policy.tokenizer([ep.target_description], padding=True,
                                       truncation=True, max_length=256, return_tensors='pt')
                mask = tok['attention_mask'].cuda()
                words, _, _ = policy.lang_model(tok['input_ids'].cuda(), mask)
                language = ((words * mask[...,None]).sum(1) / mask.sum(1,keepdim=True)).cpu().numpy()[0]
                steps = TRAIN_STEPS if split == 'train_seen' else [15]
                actual_steps = sorted({min(t, len(traj['trajectory'])-1) for t in steps})
                for t in actual_steps:
                    pose = traj['trajectory'][t]
                    image, geometry = cropper.get(ep.map_name, pose)
                    # Exactly the colour convention/normalization of HETT.
                    im = np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1), dtype='f4')
                    im = (im-policy.rgb_mean)/policy.rgb_std
                    visual = policy.vision_model(torch.from_numpy(im[None]).cuda())
                    visual = visual.reshape(512,49).T.cpu().numpy()
                    # Labels ONLY: no use in crop, state or candidate construction.
                    delta = np.array([ep.target_position.x-pose.x,
                                      pose.y-ep.target_position.y], dtype='f4')
                    all_features.append(visual); all_geo.append(geometry); all_lang.append(language)
                    labels.append(delta)
                    metadata.append(dict(id=key, map=ep.map_name, step=t,
                                         stage2_started=(t > len(traj['stage1_trajectory'])-1),
                                         pose=list(pose), instruction=ep.target_description))
            if batch_num % 16 == 0:
                print(f'COLLECT {split} {len(predictions)}/{env.size()} episodes '
                      f'{len(labels)} states {time.time()-start:.1f}s', flush=True)
        np.savez_compressed(out / f'{split}_features.npz', rgb=np.asarray(all_features),
                            geo=np.asarray(all_geo), text=np.asarray(all_lang))
        np.save(out / f'{split}_labels.npy', np.asarray(labels))
        write_json(out / f'{split}_manifest.json', dict(maps=maps, episodes=len(predictions),
                                                      samples=metadata))
        torch.save(predictions, out / f'{split}_trajectories.pt')
        print(f'DONE {split} episodes={len(predictions)} states={len(labels)} maps={maps}', flush=True)
    print('COLLECTION COMPLETE', flush=True)


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(512,64)
        self.geometry = nn.Linear(5,64,bias=False)
        self.text = nn.Linear(768,64)
        self.position = nn.Linear(4,64)
        self.score = nn.Sequential(nn.Linear(192,64), nn.ReLU(), nn.Linear(64,1))
        self.outside = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, rgb, geo, text, coords):
        vision = self.visual(rgb) + self.geometry(geo)
        pos = self.position(torch.cat([coords, coords.square()],1))[None]
        v = torch.tanh(vision + pos)
        q = torch.tanh(self.text(text))[:,None].expand_as(v)
        scores = self.score(torch.cat([v,q,v*q],-1)).squeeze(-1)
        outside = self.outside(torch.cat([v.mean(1),q[:,0]],-1))
        return torch.cat([scores,outside],1)


def metrics(errors, inside):
    result = dict(mean_error_m=float(errors.mean()), median_error_m=float(np.median(errors)))
    for radius in [5,10,20]:
        result[f'r{radius}'] = float((errors<=radius).mean()*100)
        result[f'inside_r{radius}'] = float((errors[inside]<=radius).mean()*100) if inside.any() else None
    return result


def fit(out, exclude_validation_manifest=None):
    torch.set_num_threads(2)
    write_json(out/'fit_protocol.json',dict(script_sha256=sha(Path(__file__)),
               seeds=SEEDS,epochs=30,selection='last epoch only; no validation tuning',
               primary='all-sample R@20 with outside->stay fallback',
               secondary='forced local argmax, also report GT-inside subset as diagnostic only',
               outside_decision='P(outside)>0.5; otherwise argmax among 49 local candidates',
               exclude_validation_manifest=str(exclude_validation_manifest)))
    train = dict(np.load(out / 'train_seen_features.npz'))
    test = dict(np.load(out / 'val_unseen_features.npz'))
    yt = np.load(out / 'train_seen_labels.npy')
    yv = np.load(out / 'val_unseen_labels.npy')
    meta = json.loads((out / 'val_unseen_manifest.json').read_text())
    trainmeta = json.loads((out / 'train_seen_manifest.json').read_text())
    assert not set(meta['maps']) & set(trainmeta['maps'])
    excluded = set()
    if exclude_validation_manifest:
        excluded = {s['id'] for s in json.loads(exclude_validation_manifest.read_text())['samples']}
        keep = np.array([s['id'] not in excluded for s in meta['samples']])
        test = {k:v[keep] for k,v in test.items()}
        yv = yv[keep]
        meta['samples'] = [s for s in meta['samples'] if s['id'] not in excluded]
        meta['episodes'] = len(meta['samples'])
    write_json(out/'evaluation_manifest.json',meta)
    np.save(out/'evaluation_labels.npy',yv)
    inside_t = (np.abs(yt) < SIDE/2).all(1)
    inside_v = (np.abs(yv) < SIDE/2).all(1)
    target = np.floor((yt/SIDE+.5)*GRID).astype(int)
    target = np.where(inside_t, target[:,1]*GRID+target[:,0], GRID*GRID)
    xx, yy = np.meshgrid((np.arange(GRID)+.5)/GRID-.5, (np.arange(GRID)+.5)/GRID-.5)
    coords = torch.tensor(np.stack([xx.ravel(),yy.ravel()],1), dtype=torch.float32)
    candidates = np.concatenate([coords.numpy()*SIDE, np.zeros((1,2))])
    # Feature normalization fitted ONLY on training data.
    normalization = {}
    for k in train:
        axes = tuple(range(train[k].ndim-1))
        mean, std = train[k].mean(axis=axes), train[k].std(axis=axes)
        std = np.maximum(std, .05)
        normalization[k+'_mean'] = mean; normalization[k+'_std'] = std
        train[k] = torch.tensor((train[k]-mean)/std)
        test[k] = torch.tensor((test[k]-mean)/std)
    np.savez(out/'normalization.npz', **normalization)
    target = torch.tensor(target, dtype=torch.long)
    results, raw = {}, {}
    center_error = np.linalg.norm(yv,axis=1)
    results['stay_at_observation'] = metrics(center_error, inside_v)
    raw['stay_at_observation'] = center_error
    training_log = []
    for mode in ['position_text','rgb_text','rgb_geo_text']:
        seed_errors = []
        for seed in SEEDS:
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = Localizer()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            tr_rgb = train['rgb'] if mode != 'position_text' else torch.zeros_like(train['rgb'])
            tr_geo = train['geo'] if mode == 'rgb_geo_text' else torch.zeros_like(train['geo'])
            ev_rgb = test['rgb'] if mode != 'position_text' else torch.zeros_like(test['rgb'])
            ev_geo = test['geo'] if mode == 'rgb_geo_text' else torch.zeros_like(test['geo'])
            for epoch in range(30):
                model.train(); loss_sum = 0
                for ids in np.array_split(rng.permutation(len(yt)), max(1,int(np.ceil(len(yt)/32)))):
                    logits = model(tr_rgb[ids], tr_geo[ids], train['text'][ids], coords)
                    loss = F.cross_entropy(logits,target[ids])
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5)
                    optimizer.step(); loss_sum += loss.item()*len(ids)
                training_log.append(dict(mode=mode,seed=seed,epoch=epoch+1,loss=loss_sum/len(yt)))
            model.eval()
            torch.save(model.state_dict(),out/f'{mode}_seed{seed}.pt')
            with torch.no_grad():
                cases = {'normal':(ev_rgb,ev_geo,test['text'])}
                perm = np.random.default_rng(123).permutation(len(yv))
                cases['shuffled_text'] = (ev_rgb,ev_geo,test['text'][perm])
                if mode == 'rgb_geo_text':
                    gp = np.random.default_rng(123).permutation(49)
                    cases['shuffled_geometry'] = (ev_rgb,ev_geo[:,gp],test['text'])
                for condition, (rv,gv,tv) in cases.items():
                    logits = model(rv,gv,tv,coords)
                    # Compare outside to the total local probability mass, not
                    # to each of 49 fragmented local hypotheses separately.
                    outside = logits.softmax(1)[:,-1] > .5
                    pred = logits[:,:49].argmax(1)
                    pred[outside] = 49
                    pred = pred.numpy()
                    error = np.linalg.norm(candidates[pred]-yv,axis=1)
                    key = f'{mode}/{condition}/seed{seed}'
                    raw[key] = error
                    raw[key+'/prediction'] = candidates[pred]
                    results[key] = metrics(error, inside_v)
                    results[key]['abstain_pct'] = float((pred==49).mean()*100)
                    forced = logits[:,:49].argmax(1).numpy()
                    forced_error = np.linalg.norm(candidates[forced]-yv,axis=1)
                    results[key]['forced_local'] = metrics(forced_error, inside_v)
                    raw[key+'/forced_prediction'] = candidates[forced]
                    if condition == 'normal': seed_errors.append(error)
            print('FIT',mode,seed,results[f'{mode}/normal/seed{seed}'],flush=True)
        errors = np.stack(seed_errors)
        results[mode] = dict(r20_seed_mean=float((errors<=20).mean()*100),
                             r20_seed_values=((errors<=20).mean(1)*100).tolist(),
                             inside_r20_seed_mean=float((errors[:,inside_v]<=20).mean()*100))
    # Paired bootstrap over MAPS, not independent correlated descriptions.
    baseline = np.stack([raw[f'rgb_text/normal/seed{s}']<=20 for s in SEEDS]).mean(0)
    enhanced = np.stack([raw[f'rgb_geo_text/normal/seed{s}']<=20 for s in SEEDS]).mean(0)
    groups = np.array([item['map'] for item in meta['samples']])
    unique = np.unique(groups)
    rng = np.random.default_rng(777)
    boot = []
    for _ in range(5000):
        idx = np.concatenate([np.flatnonzero(groups==m) for m in rng.choice(unique,len(unique),replace=True)])
        boot.append(float((enhanced[idx]-baseline[idx]).mean()*100))
    coverage = dict(train_states=len(yt),train_episodes=trainmeta['episodes'],val_states=len(yv),
                    train_inside_pct=float(inside_t.mean()*100),val_inside_pct=float(inside_v.mean()*100),
                    validation_maps=unique.tolist(),
                    smoke_validation_ids_excluded=sorted(excluded),
                    paired_geo_minus_rgb_r20_pp=float((enhanced-baseline).mean()*100),
                    map_bootstrap_95ci_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                    caveat='Exploratory fixed epoch-1 features; few held-out maps; not closed-loop navigation.')
    write_json(out/'metrics.json',dict(coverage=coverage,results=results))
    write_json(out/'training_losses.json',training_log)
    np.savez_compressed(out/'predictions.npz',**raw)
    print('SUMMARY',json.dumps(coverage),flush=True)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    report=json.loads((out/'metrics.json').read_text()); results=report['results']
    data=dict(np.load(out/'predictions.npz'))
    labels=['原地不改','位置＋语言','RGB＋语言','RGB＋几何＋语言']
    modes=['position_text','rgb_text','rgb_geo_text']
    fig, ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    means=[results['stay_at_observation']['r20']]+[results[m]['r20_seed_mean'] for m in modes]
    ax[0,0].bar(labels,means,color=['#aaa','#aaa','#3388bb','#ee9933'])
    for i,m in enumerate(modes,1):
        ax[0,0].scatter([i]*3,results[m]['r20_seed_values'],c='black',s=20)
    for i,v in enumerate(means):ax[0,0].text(i,v+.5,f'{v:.1f}%',ha='center')
    ax[0,0].set(title='全部验证样本：定位 R@20m（点为三个训练种子）',ylabel='比例（%）',ylim=(0,max(means)+10))
    for mode,label in zip(modes,labels[1:]):
        errs=np.stack([data[f'{mode}/normal/seed{s}'] for s in SEEDS])
        thresholds=np.linspace(0,60,121)
        ax[0,1].plot(thresholds,[(errs<=t).mean()*100 for t in thresholds],label=label)
    ax[0,1].legend();ax[0,1].set(title='定位误差累计分布',xlabel='误差阈值（米）',ylabel='命中率（%）')
    conditions=['normal','shuffled_text','shuffled_geometry']
    scores=[np.mean([results[f'rgb_geo_text/{c}/seed{s}']['r20'] for s in SEEDS]) for c in conditions]
    ax[1,0].bar(['正常输入','打乱语言','打乱几何位置'],scores,color=['#ee9933','#aa7777','#7777aa'])
    for i,v in enumerate(scores):ax[1,0].text(i,v+.5,f'{v:.1f}%',ha='center')
    ax[1,0].set(title='输入依赖检查：增强模型',ylabel='R@20m（%）',ylim=(0,max(scores)+10))
    loss=json.loads((out/'training_losses.json').read_text())
    for mode,label in zip(modes,labels[1:]):
        values=[[r['loss'] for r in loss if r['mode']==mode and r['seed']==s] for s in SEEDS]
        ax[1,1].plot(np.arange(1,31),np.mean(values,0),label=label)
    ax[1,1].legend(); ax[1,1].set(title='训练损失（三种子均值，不用验证挑轮次）',xlabel='训练轮次',ylabel='交叉熵')
    fig.suptitle('局部点云几何离线测试｜真实学生轨迹、未见地图、非导航成功率',fontsize=15)
    fig.savefig(out/'comparison.png',dpi=160)
    fig.savefig(out/'comparison.pdf');plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--phase',choices=['collect','fit','plot'],required=True)
    p.add_argument('--train',type=int,default=512)
    p.add_argument('--val',type=int,default=256)
    p.add_argument('--exclude-validation-manifest',type=Path)
    opts=p.parse_args(); opts.out.mkdir(parents=True,exist_ok=True)
    if opts.phase=='collect':
        protocol=dict(checkpoint=str(opts.checkpoint),checkpoint_sha256=sha(opts.checkpoint),
                      script_sha256=sha(Path(__file__)),train=opts.train,val=opts.val,
                      train_steps=TRAIN_STEPS,val_step=15,side_m=SIDE,grid=GRID,seeds=SEEDS,
                      epochs=30,lr=.001,batch=32,world_coords_input=False,semantic_labels_input=False,
                      model='frozen HETT epoch-1 encoders + small local candidate classifier',
                      scope='current-view-contained north-up ortho patch, no occlusion simulator',
                      evaluation='localization only; outside prediction falls back to observation position')
        write_json(opts.out/'protocol.json',protocol)
        collect(opts.out,opts.checkpoint,opts.train,opts.val)
    elif opts.phase=='fit':fit(opts.out,opts.exclude_validation_manifest)
    else:plot(opts.out)
