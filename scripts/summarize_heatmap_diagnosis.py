#!/usr/bin/env python3
"""Summarize completed diagnostic and full-split evaluations without rerunning."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
def load(p):return json.loads(p.read_text())
def main():
 for name in ['candidate_controller','full_validation','handoff_control']:
  if not (ART/name/'COMPLETE.json').exists():raise RuntimeError(f'{name} is incomplete')
 diag=load(ART/'candidate_controller/results.json');full=load(ART/'full_validation/results.json');handoff=load(ART/'handoff_control/results.json')
 assert len(diag)==22 and len(full)==4 and len(handoff)==4
 for split,n in [('val_seen',2470),('val_unseen',2697)]:
  rows=[r for r in full if r['split']==split];assert all(r['episodes']==n for r in rows)
  base=next(r for r in rows if r['variant']=='baseline')
  assert round(base['sr'],2)==({'val_seen':21.94,'val_unseen':16.65}[split])
  a=load(ART/'candidate_controller'/f'{split}_direct_bounded.json')['episodes']
  b=load(ART/'candidate_controller'/f'{split}_waypoint_api.json')['episodes']
  assert a==b
 report={'diagnostic_runs':diag,'full_validation':full,'handoff_control':handoff,
  'sbf_candidate_comparison':load(ART/'candidate_controller/sbf_candidate_comparison.json'),
  'failure_analysis':load(ART/'candidate_controller/failure_analysis.json'),
  'checks':{'unit_tests_passed':28,'full_baseline_matches_original_rounded_metrics':True,'deployable_api_matches_diagnostic_traces':True,'test_unseen_used':False,'training_restarted':False},
  'limitations':['Oracle variants use GT only for diagnosis, never a deployable score.','SBF comparison uses epoch14, HETT uses epoch1; this is not equal training budget.','Removing a gate at inference measures learned dependence, not retrained module attribution.','Direct execution changes fine-control policy, stopping and horizontal kinematics; direct_discrete separately retains old low-level actions.','Validation was used for diagnosis; results do not constitute a held-out Test-Unseen estimate.','Paired bootstrap intervals resample episodes and do not account for shared scenes/objects.']}
 dest=ART/'diagnosis_summary';dest.mkdir(exist_ok=True)
 (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 fields=['scope','split','variant','episodes','sr','oracle_sr','spl','ne','delta_sr_pp','gained','lost','privileged']
 with (dest/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for scope,rows in [('subset256',diag),('full',full),('handoff256',handoff)]:
   for row in rows:w.writerow({k:scope if k=='scope' else row.get(k) for k in fields})
 lines=['HETT HEATMAP CANDIDATE / CONTROLLER DIAGNOSIS','',
 'Frozen checkpoint: epoch 1, best_val_unseen from heatmap_relative_geometry_top20_688e981_b8_s0_2e_20261008.',
 'No training, no Test-Unseen. Original worktree and checkpoints preserved.','',
 'MAIN FINDING: The two-stage execution path loses much of the benefit of correct proposals. Candidate ranking remains a separate bottleneck.',
 'The multi-landmark branch is already active; disabling it sharply degrades both candidate recall and navigation.','',
 'FULL VALIDATION (SR, OSR, SPL are percentages; NE is meters):']
 for r in full:lines.append(f"{r['split']} {r['variant']}: N={r['episodes']}, SR={r['sr']:.4f}, OSR={r['oracle_sr']:.4f}, SPL={r['spl']:.4f}, NE={r['ne']:.4f}, delta SR={r['delta_sr_pp']:.4f} pp, paired CI95={r['delta_sr_ci95']}")
 lines+=['','FIX: --mode eval --heatmap_execution waypoint',
 'Default is still two_stage. The new option is inference-only.',
 'Keep the frozen heatmap Top-1 and its local soft-argmax target; move at most 50m without overshooting the waypoint; preserve altitude; replan each step; stop after 5 stationary steps or the 20-step horizon.',
 'Do not use learned fine direction/progress heads to redirect this waypoint controller. No GT enters the movement or termination rule.',
 'No claim is made that this resolves incorrect heatmap rankings or all stopping failures.','',
 'REPRODUCTION (from this worktree; HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):',
 'python scripts/diagnose_candidate_controller.py',
 'python scripts/diagnose_candidate_controller.py --config <absolute>/configs/diagnostics/full_validation.json --output <fresh absolute output> --full',
 'python scripts/diagnose_candidate_controller.py --config <absolute>/configs/diagnostics/handoff_control.json --output <fresh absolute output>',
 'python -m unittest tests.test_heatmap_execution tests.test_heatmap_relative_geometry tests.test_multi_landmark_relations tests.test_dense_spatial_belief -q','',
 'Measurement fixes: Top16 distance now slices the first 16 candidates; Stage2 entry distance uses obs[i] instead of a stale batch variable. These do not change baseline decisions.','',*report['limitations']]
 (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
