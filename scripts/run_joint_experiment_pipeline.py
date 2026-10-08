#!/usr/bin/env python3
"""Run the user's one-epoch budget, then commit and push compact deliverables."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_goal_trajectory/run1'
BRANCH='2027-CVPR/heatmap-joint-goal-trajectory'

def main():
    cfg=ROOT/'configs/experiments/joint_goal_trajectory_1e.json'
    if json.loads(cfg.read_text())['epochs']!=1:raise RuntimeError('User capped this run at one epoch')
    smoke=ROOT/'artifacts/joint_goal_trajectory/smoke/COMPLETE.json'
    if not smoke.exists():raise RuntimeError('GPU smoke must finish first')
    OUT.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run_joint_goal_trajectory_experiment.py'),
         '--config',str(cfg),'--output',str(OUT)]
    with (OUT/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    done=json.loads((OUT/'COMPLETE.json').read_text())
    if done['epochs']!=1 or done['eval_runs']!=8:raise RuntimeError('Incomplete experiment')
    if not (OUT/'figures/index.json').exists():raise RuntimeError('Missing trajectory figures')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH:raise RuntimeError('Branch changed; refusing automatic commit')
    subprocess.run(['git','add','artifacts/joint_goal_trajectory/run1','docs/HEATMAP_JOINT_GOAL_TRAJECTORY_EXPERIMENT.md'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Report matched one-epoch joint goal/trajectory experiment and paired failures'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_TERMINAL_PROMPT='0')
    result=subprocess.run(['git','-c','credential.helper=!gh auth git-credential','push',
        'https://github.com/qiaoxu123/HETT.git',f'HEAD:{BRANCH}'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    status={'commit':commit,'branch':branch,'push_returncode':result.returncode,'time':time.time(),
            'push_output':result.stdout+result.stderr}
    (OUT/'delivery_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    if result.returncode:raise RuntimeError('Results committed; push failed. See delivery_status.json')
if __name__=='__main__':main()
