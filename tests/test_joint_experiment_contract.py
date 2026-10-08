import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class JointExperimentContracts(unittest.TestCase):
    def test_config_fixed_pairs(self):
        c=json.loads((ROOT/'configs/experiments/joint_goal_trajectory_1e.json').read_text())
        self.assertEqual(c['epochs'],1);self.assertEqual(c['evaluation_splits'],['val_seen','val_unseen'])
        self.assertEqual(set(c['variants']),set('ABCD'))
        b=c['variants']['B'];cc=c['variants']['C'];d=c['variants']['D']
        self.assertEqual({k for k in b if b[k]!=cc[k]},{'trajectory_selector_mode'})
        self.assertEqual({k for k in cc if cc[k]!=d[k]},{'trajectory_disable_learned_stop'})
    def test_diagnostic_only_instrumentation(self):
        s=(ROOT/'multiagent/agent.py').read_text()
        self.assertLess(s.index('# Interact with the simulator with actions'),s.index('self.experiment_step_callback('))
        self.assertNotIn('test_unseen', (ROOT/'scripts/run_joint_goal_trajectory_experiment.py').read_text())
    def test_no_model_architecture_changes(self):
        import subprocess
        files=subprocess.check_output(['git','diff','origin/2027-CVPR/heatmap-joint-goal-trajectory','--name-only','--','multiagent/models'],cwd=ROOT,text=True).strip()
        self.assertEqual(files,'')
if __name__=='__main__':unittest.main()
