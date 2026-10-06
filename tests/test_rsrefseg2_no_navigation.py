from pathlib import Path
def test_experiment_scripts_do_not_import_controller_or_rollout():
 root=Path(__file__).parents[1]
 text='\n'.join(p.read_text() for p in (root/'scripts').glob('*rsrefseg2*py'))
 assert 'multiagent.agent' not in text and 'two_stage' not in text.casefold() and 'navigation rollout' not in text.casefold()
