"""Contracts for frozen transfer; full GPU inference covered by smoke episodes."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class TransferContracts(unittest.TestCase):
    def test_choose_api_has_no_labels(self):
        tree=ast.parse((ROOT/'multiagent/models/frozen_sbf_selector.py').read_text())
        method=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='choose')
        self.assertEqual({x.arg for x in method.args.kwonlyargs},
            {'episode_ids','instructions','histories','candidate_world_xy','split','active'})
    def test_builder_record_excludes_target_and_teacher(self):
        tree=ast.parse((ROOT/'multiagent/models/frozen_sbf_selector.py').read_text())
        record=next(x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='SimpleNamespace')
        self.assertEqual({x.arg for x in record.keywords},
            {'map_name','object_id','description_id','episode_id','instruction'})
    def test_paired_config(self):
        import json
        c=json.loads((ROOT/'configs/diagnostics/frozen_sbf_selector.json').read_text())
        self.assertEqual(c['splits'],['val_seen','val_unseen'])
        self.assertEqual(c['variants']['baseline']['execution'],c['variants']['sbf_selector']['execution'])
        self.assertFalse(any(v.get('oracle_candidate') or v.get('oracle_stop') for v in c['variants'].values()))
if __name__=='__main__':unittest.main()
