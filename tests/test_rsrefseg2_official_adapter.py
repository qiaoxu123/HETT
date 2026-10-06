from pathlib import Path
import inspect
from multiagent.rsrefseg2_grounding.official_adapter import OFFICIAL_COMMIT,OfficialCoarseGrounder,load_official_prompter_class
def test_official_prompter_loaded_from_pinned_source():
 source=Path('/mnt/windows-data/external/RSRefSeg2/refseg/models/models.py')
 if source.exists():assert load_official_prompter_class(source).__name__=='CascadedPrompter'
 assert len(OFFICIAL_COMMIT)==40
def test_text_padding_is_batch_invariant_by_construction():
 source=inspect.getsource(OfficialCoarseGrounder.forward)
 assert 'padding="max_length"' in source and 'max_length=64' in source
