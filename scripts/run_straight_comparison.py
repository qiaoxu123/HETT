"""Serial paired single-seed screen. Existing worktrees remain read-only."""
import json,subprocess,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PARENT=ROOT.parent/'01-teacher-fix/runs/teacher_fix_full_s0/checkpoints/latest'
RUN=ROOT/'runs/paired_screen_s0'
def save(name,value):
    path=RUN/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def main():
    RUN.mkdir(parents=True,exist_ok=False)
    save('protocol.json',{'seed':0,'parent':str(PARENT),'parent_epoch':7,'additional_epochs':1,'sample_size_per_split':512,
       'sampling':'fixed seed0 per-map shuffled round robin, identical both arms','teacher_modes':['human','straight'],
       'changes':'straight oracle path only for teacher movement on train_seen; original reference and eval trajectories preserved',
       'limitations':'short fine-tuning screen, not from-scratch; trajectories/observations differ intentionally; single seed; no test_unseen',
       'primary':'paired val_unseen SR (20m), secondary NE/SPL; >=3pp SR gain is preliminary practical signal, not significance'})
    for mode in ['human','straight']:
        save('status.json',{'phase':mode,'time':time.time()})
        command=[PY,'-u',str(ROOT/'scripts/supervise_experiment.py'),'--python',PY,'--run-dir',str(RUN/mode),
           '--epochs','8','--max-episodes','512','--seed','0','--save-every','1','--interval','30',
           '--variant-arg=--disable_task_interaction','--variant-arg=--balanced_screen',
           '--train-variant-arg=--checkpoint',f'--train-variant-arg={PARENT}',
           '--train-variant-arg=--teacher_path_mode',f'--train-variant-arg={mode}']
        with (RUN/(mode+'_supervisor.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save('status.json',{'phase':'failed','arm':mode,'exit_code':result.returncode});raise SystemExit(result.returncode)
    result=subprocess.run([PY,str(ROOT/'scripts/report_straight_comparison.py')],cwd=ROOT)
    save('status.json',{'phase':'complete' if result.returncode==0 else 'report_failed','time':time.time()})
if __name__=='__main__':main()
