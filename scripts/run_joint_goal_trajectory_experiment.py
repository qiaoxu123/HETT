#!/usr/bin/env python3
"""Matched fixed-budget experiment; model architecture and action rules stay upstream."""
import os,sys,json,time,random,hashlib,argparse,subprocess,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'multiagent'),str(ROOT),str(ROOT/'scripts')]
os.chdir(ROOT/'multiagent')
from agent import NavCMTAgent
from env import CityNavBatch
from parser import parse_args
from torch.utils.data import DataLoader
from joint_experiment_metrics import observe,summarize


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.generic,)):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def strict_init(agent,path):
    data=torch.load(path,map_location='cpu',weights_only=False);report={}
    for key,model in [('lang_model',agent.lang_model_without_ddp),('vision_model',agent.vision_model_without_ddp),('vln_model',agent.vln_model_without_ddp)]:
        state=data[key]['state_dict'];missing,extra=model.load_state_dict(state,strict=False)
        if extra or any(not k.startswith('trajectory_head.') for k in missing):
            raise RuntimeError(f'Unexpected checkpoint mismatch {key}: {missing}, {extra}')
        report[key]={'missing_new_parameters':missing,'unexpected':extra,'loaded_tensors':len(state)}
    return report

class GradientAudit:
    def __init__(self):self.rows=[]
    def __call__(self,agent,batch):
        if not torch.isfinite(agent.loss.detach()):raise RuntimeError('nonfinite loss')
        if batch>2:return
        groups=defaultdict(list)
        for n,p in agent.vln_model_without_ddp.named_parameters():
            if p.grad is None:continue
            group='base'
            if n.startswith('trajectory_head.'):
                group='trajectory.'+n.split('.')[1]
            groups[group].append(p.grad.detach().float().norm())
        norms={k:float(torch.stack(v).norm()) for k,v in groups.items()}
        if not all(math.isfinite(v) for v in norms.values()):raise RuntimeError('nonfinite gradients')
        self.rows.append({'batch':batch,'loss':float(agent.loss.detach()),'gradient_norms':norms})

def new_agent(args,cfg):
    seed(cfg['seed']);agent=NavCMTAgent(args,allow_ngpus=False,rank=0)
    report=strict_init(agent,cfg['initial_checkpoint'])
    def inference_guard(module,inputs,kwargs):
        if not module.training and kwargs.get('trajectory_teacher_goal') is not None:
            raise RuntimeError('GT-conditioned forward forbidden in evaluation')
    agent.vln_model_without_ddp.register_forward_pre_hook(inference_guard,with_kwargs=True)
    return agent,report

def evaluate(agent,envs,cfg,out,epoch):
    rows=[];agent.experiment_step_callback=observe
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    for split,env in envs.items():
        for variant,flags in cfg['variants'].items():
            seed(cfg['seed'])
            for k,v in flags.items():setattr(agent.args,k,v)
            agent.env=env;agent.logs=defaultdict(list)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            agent.test(DataLoader(env,batch_size=1),env_name=split,feedback='student')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            r=summarize(env,agent.get_results(),variant,epoch,seconds)
            r['summary'].update(peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
            dump(out/f'epoch{epoch:02d}'/f'{split}_{variant}.json',r)
            rows.append(r['summary']);print('EVAL_COMPLETE',json.dumps(clean(r['summary'])),flush=True)
    agent.experiment_step_callback=None
    random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
    for k,v in cfg['variants']['A'].items():setattr(agent.args,k,v)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    opt=ap.parse_args();cfg=json.loads(opt.config.read_text());out=opt.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Use a new output; existing experiment will not be overwritten')
    if cfg['evaluation_splits']!=['val_seen','val_unseen']:raise ValueError('Development splits must be seen/unseen validation only')
    sys.argv=['joint_experiment'];args=parse_args()
    # Preserve optimizer/controller/data settings of the declared common initialization experiment.
    for k,v in json.loads(Path(cfg['base_arguments']).read_text()).items():
        if hasattr(args,k):setattr(args,k,v)
    args.mode='train';args.resume_optimizer=False;args.batch_size=cfg['batch_size'];args.seed=cfg['seed'];args.benchmark_batches=0
    args.epochs=cfg['epochs'];args.heatmap_trajectory_enabled=True;args.trajectory_use_for_control=False
    for k,v in cfg.items():
        if k.startswith('trajectory_'):setattr(args,k,v)
    args.trajectory_selector_mode='prior';args.trajectory_disable_learned_stop=False
    dump(out/'manifest.json',dict(config=cfg,args=vars(args),smoke=opt.smoke,initial_sha256=sha(cfg['initial_checkpoint']),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started=time.time(),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'multiagent/agent.py',ROOT/'multiagent/models/heatmap_trajectory.py',Path(__file__),ROOT/'scripts/joint_experiment_metrics.py']}))
    agent,loadreport=new_agent(args,cfg);dump(out/'checkpoint_load.json',loadreport)
    start=time.time();train=CityNavBatch('train_seen',args,batch_size=args.batch_size,seed=args.seed,rank=0,world_size=1)
    envs={s:CityNavBatch(s,args,batch_size=args.batch_size,seed=args.seed,rank=0,world_size=1) for s in cfg['evaluation_splits']}
    dump(out/'data.json',{'load_seconds':time.time()-start,'train_count':len(train.data),'validation_counts':{s:len(e.data) for s,e in envs.items()},
        'annotation_sha256':{s:sha(ROOT/'data/processed_citynav'/f'citynav_{s}.json') for s in ['train_seen',*cfg['evaluation_splits']]}})
    for s,e in envs.items():dump(out/'episode_ids'/f'{s}.json',[list(x.id) for x in e.data])
    if opt.smoke:
        for e in envs.values():e.data=e.data[:8]
        for batch in (2,8):
            # Fresh weights/optimizer for each disposable GPU smoke configuration.
            if batch!=2:
                del agent
                import gc
                gc.collect();torch.cuda.empty_cache();agent,loadreport=new_agent(args,cfg)
            args.batch_size=batch;train.batch_size=batch;args.benchmark_batches=2
            agent.env=train;agent.env_name='train_seen';agent.logs=defaultdict(list)
            audit=GradientAudit();agent.experiment_gradient_callback=audit
            before={k:v.detach().clone() for k,v in agent.vln_model_without_ddp.trajectory_head.named_parameters()}
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            agent.train(DataLoader(train,batch_size=1),1,feedback='student')
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            delta={k:float((v.detach()-before[k]).norm()) for k,v in agent.vln_model_without_ddp.trajectory_head.named_parameters()}
            if not all(math.isfinite(x) for x in delta.values()) or not any(x>0 for x in delta.values()):raise RuntimeError('Missing/nonfinite head updates')
            r=dict(batch_size=batch,optimizer_steps=2,seconds=elapsed,seconds_per_batch=elapsed/2,
                peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
                gradients=audit.rows,head_update_norms=delta,losses={k:float(np.mean(v)) for k,v in agent.logs.items() if 'loss' in k and len(v)})
            dump(out/f'gpu_batch{batch}.json',r);print('GPU_SMOKE',json.dumps(r),flush=True)
        args.benchmark_batches=0
        rows=evaluate(agent,envs,cfg,out,0);dump(out/'results.json',rows)
        dump(out/'COMPLETE.json',{'smoke':True,'time':time.time(),'eval_runs':len(rows)})
        return
    (out/'checkpoints').mkdir(exist_ok=True)
    agent.save(-1,str(out/'checkpoints'/'initial.pt'))
    allrows=[]
    for epoch in range(1,cfg['epochs']+1):
        agent.env=train;agent.env_name='train_seen';agent.logs=defaultdict(list);agent.experiment_step_callback=None
        audit=GradientAudit();agent.experiment_gradient_callback=audit
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        agent.train(DataLoader(train,batch_size=1),1,feedback='student')
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        record=dict(epoch=epoch,train_seconds=elapsed,episodes=len(train.data),batches=math.ceil(len(train.data)/train.batch_size),
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),gradients=audit.rows,
            losses={k:float(np.mean(v)) for k,v in agent.logs.items() if 'loss' in k and len(v)})
        if not all(math.isfinite(v) for v in record['losses'].values()):raise RuntimeError('Nonfinite epoch loss')
        ckpt=out/'checkpoints'/f'epoch{epoch:02d}.pt';tmp=ckpt.with_suffix('.tmp')
        agent.save(epoch-1,str(tmp));tmp.replace(ckpt)
        torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},out/'checkpoints'/f'epoch{epoch:02d}_rng.pt')
        record.update(checkpoint=str(ckpt),sha256=sha(ckpt))
        dump(out/f'train_epoch{epoch:02d}.json',record);print('TRAIN_COMPLETE',json.dumps(clean(record)),flush=True)
        allrows.extend(evaluate(agent,envs,cfg,out,epoch));dump(out/'results.json',allrows)
        subprocess.run([sys.executable,str(ROOT/'scripts/report_joint_goal_trajectory.py'),str(out)],check=True,cwd=ROOT)
    dump(out/'COMPLETE.json',{'epochs':cfg['epochs'],'eval_runs':len(allrows),'time':time.time()})
    subprocess.run([sys.executable,str(ROOT/'scripts/report_joint_goal_trajectory.py'),str(out)],check=True,cwd=ROOT)
if __name__=='__main__':main()
