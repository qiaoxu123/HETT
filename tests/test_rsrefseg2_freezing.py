import torch
from multiagent.rsrefseg2_grounding.official_adapter import configure_trainable
class M(torch.nn.Module):
 def __init__(self):
  super().__init__();self.prompter=torch.nn.Linear(2,2);self.clip_vision_encoder=torch.nn.Module();self.clip_vision_encoder.lora_A=torch.nn.Linear(2,1);self.clip_text_encoder=torch.nn.Linear(2,2)
def test_text_always_frozen():
 for mode in ('prompter','vision_lora'):
  m=M();configure_trainable(m,mode);assert not any(p.requires_grad for p in m.clip_text_encoder.parameters())
