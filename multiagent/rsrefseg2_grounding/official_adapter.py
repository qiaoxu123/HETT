"""Thin adapter around the official RSRefSeg2 cascaded prompter.

The prompter class definitions are loaded directly from an audited checkout of
KyanChen/RSRefSeg2 (Apache-2.0) rather than rewritten.  OpenMMLab is not needed
for the SAM-before coarse map used by this diagnostic.
"""
from __future__ import annotations
import ast,math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional,Tuple
import torch
from torch import Tensor,nn
from torch.nn import functional as F

OFFICIAL_COMMIT="b717f8dbbd9dbb67cb711e71cff49e03ba53c258"
CLASSES=("Attention","MLP","PITCrossAttentionBlock","TwoQueryTextCrossAttentionBlock","CascadedPrompter")

def load_official_prompter_class(source:Path):
    import einops
    class BaseModule(nn.Module):
        def __init__(self,init_cfg=None):super().__init__();self.init_cfg=init_cfg
    tree=ast.parse(Path(source).read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in CLASSES:
            node.decorator_list=[]
            if node.name=="CascadedPrompter":node.bases=[ast.Name(id="BaseModule",ctx=ast.Load())]
            nodes.append(node)
    if {n.name for n in nodes}!=set(CLASSES):raise RuntimeError("official RSRefSeg2 source layout changed")
    module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module);namespace={"math":math,"Optional":Optional,"Tuple":Tuple,"einops":einops,"torch":torch,"Tensor":Tensor,"nn":nn,"F":F,"BaseModule":BaseModule}
    exec(compile(module,str(source),"exec"),namespace)
    return namespace["CascadedPrompter"]

class VisionWrapper(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,*a,**kw):return self.model(*a,**kw)
class TextWrapper(nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.logit_scale=nn.Parameter(torch.ones(1)*math.log(10));self.logit_bias=nn.Parameter(torch.zeros(1))
    def forward(self,*a,**kw):return self.model(*a,**kw)

class OfficialCoarseGrounder(nn.Module):
    """Official SigLIP2 encoders + CascadedPrompter, stopping before SAM."""
    def __init__(self,official_root:Path,model_name="google/siglip2-so400m-patch16-512",lora_rank=16):
        super().__init__()
        from transformers import AutoConfig,AutoProcessor,SiglipTextModel,SiglipVisionModel
        from peft import LoraConfig,get_peft_model
        config=AutoConfig.from_pretrained(model_name);self.processor=AutoProcessor.from_pretrained(model_name);vision=VisionWrapper(SiglipVisionModel(config.vision_config).vision_model);text=TextWrapper(SiglipTextModel(config.text_config).text_model)
        vision_target=r"^(model\.head\.(attention\.out_proj|mlp\.fc[12])|model\.encoder\.layers\.\d+\.self_attn\.(k_proj|v_proj|q_proj|out_proj))$"
        text_target=r"^(model\.head|model\.encoder\.layers\.\d+\.self_attn\.(k_proj|v_proj|q_proj|out_proj))$"
        self.clip_vision_encoder=get_peft_model(vision,LoraConfig(r=lora_rank,lora_alpha=16,lora_dropout=0.,target_modules=vision_target));self.clip_text_encoder=get_peft_model(text,LoraConfig(r=lora_rank,lora_alpha=16,lora_dropout=0.,target_modules=text_target))
        Prompter=load_official_prompter_class(Path(official_root)/"refseg/models/models.py")
        self.prompter=Prompter(num_text_queries=3,text_query_config={"has_pe":True,"init_type":"textpool"},prompt_config={"has_pe":True,"init_type":"learnable"},text_feat_dim=1152,two_queries_text_attn_depth=2,two_queries_text_attn_operation_order=['self_attn_query_query1','norm_query1_1','self_attn_query_query2','norm_query2_1','cross_attn_query_text_query1_text','norm_query1_2','mlp_query1_2','norm_query1_3','cross_attn_query_text_query2_text','norm_query2_2','mlp_query2_2','norm_query2_3','cross_attn_query_query_query1_query2','norm_query1_4','mlp_query1_3','norm_query1_5','cross_attn_query_query_query2_query1','norm_query2_4','mlp_query2_3','norm_query2_5'],imgpe_config={"type":"learnable","size":32},queries1_to_img_attn_depth=2,queries1_to_img_attn_operation_order=['self_attn_text_text_1','norm_text_1','cross_attn_img_text_1','norm_img_1','mlp_img_1','norm_img_2'],prompt_queries2_img_attn_depth=2,prompt_feat_dim=256,prompt_queries2_img_attn_operation_order=['self_attn_text_1','norm_text_1','cross_attn_img_text_1','norm_img_1','mlp_img_1','norm_img_2','self_attn_prompt_1','norm_prompt_1','cross_attn_prompt_text_1','norm_prompt_2','mlp_prompt_2','norm_prompt_3','cross_attn_prompt_img_1','norm_prompt_4','mlp_prompt_3','norm_prompt_5'],num_prompts=9,dense_prompt_config={"proj_from":"text_pool","has_internal_proj":False,"featmap_from":"refined_clipfeat","has_internal_conv":True,"up_strategy":"preup"},img_feat_dim=1152,num_heads=8,mlp_dim=512,internal_dim=512,is_residual=True)
        self.sam_stub=SimpleNamespace(model=SimpleNamespace(sam_prompt_encoder=SimpleNamespace(mask_input_size=(256,256))))

    def load_official_checkpoint(self,path):
        state=torch.load(path,map_location="cpu",weights_only=False,mmap=True)["state_dict"]
        selected={k:v for k,v in state.items() if k.startswith(("clip_vision_encoder.","clip_text_encoder.","prompter."))}
        missing,unexpected=self.load_state_dict(selected,strict=False)
        bad=[x for x in missing if not x.startswith("sam_stub")]
        if bad or unexpected:raise RuntimeError(f"official checkpoint mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
        return len(selected)

    def forward(self,images,texts):
        # Inputs are RGB float tensors in [0,1], resized/normalized exactly as SigLIP2.
        proc=self.processor.image_processor;size=proc.size.get("height",proc.size.get("shortest_edge",512));pixels=F.interpolate(images,size=(size,size),mode="bilinear",align_corners=False);mean=torch.tensor(proc.image_mean,device=pixels.device,dtype=pixels.dtype)[None,:,None,None];std=torch.tensor(proc.image_std,device=pixels.device,dtype=pixels.dtype)[None,:,None,None];pixels=(pixels-mean)/std
        # The official prompter consumes the complete text feature sequence and
        # has no attention-mask argument of its own. Fixed-length padding makes
        # a sample invariant to the lengths of unrelated texts in its batch.
        visual=self.clip_vision_encoder(pixel_values=pixels);tokens=self.processor.tokenizer(texts,return_tensors="pt",padding="max_length",truncation=True,max_length=64);tokens={k:v.to(images.device) for k,v in tokens.items()};language=self.clip_text_encoder(**tokens)
        out=self.prompter(clip_visual_feat=visual.last_hidden_state,clip_text_feat=language.last_hidden_state,clip_text_feat_pooler=language.pooler_output,clip_text_encoder=self.clip_text_encoder.base_model.model,sam_model=self.sam_stub,output_vis_feat=False)
        return {"coarse_logits":out["dense_prompts"],"refined_features":out["refined_img_feat_map"],"vision_pool":visual.pooler_output,"text_pool":language.pooler_output}

def configure_trainable(model,mode):
    model.requires_grad_(False)
    if mode in ("prompter","vision_lora"):
        model.prompter.requires_grad_(True)
    if mode=="vision_lora":
        for name,p in model.clip_vision_encoder.named_parameters():
            if "lora_" in name:p.requires_grad_(True)
    # Text encoder remains frozen in every experiment.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
