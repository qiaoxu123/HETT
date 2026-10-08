"""Transfer the trained SBF selector with its original frozen SigLIP input path."""
from pathlib import Path
import sys
from types import SimpleNamespace
import hashlib
import numpy as np
import torch


class FrozenSBFSelector:
    def __init__(self, source, checkpoint, data_root):
        sys.path.insert(0, str(Path(source).resolve()))
        from sbfnav.model_factory import build_backbone, build_model
        from sbfnav.dataset import CityReferCatalog, CityNavDataset
        from sbfnav.planning_state import PlanningStateBuilder
        from sbfnav.runtime import TextFeatureCache
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = payload['config']
        if self.config['selector'].get('use_spatial_context', False):
            raise ValueError('This transfer targets the original epoch-14 selector')
        model = build_model(self.config)
        self.selector = model.selector
        weights = {k[len('selector.'):]: v for k,v in payload['model'].items() if k.startswith('selector.')}
        missing, unexpected = self.selector.load_state_dict(weights, strict=False)
        allowed = {k for k in self.selector.state_dict() if k.startswith(('spatial_encoder.', 'spatial_norm.')) or k == 'spatial_gate'}
        if unexpected or set(missing) != allowed:
            raise ValueError(f'Unexpected selector checkpoint schema: {missing}, {unexpected}')
        self.selector.cuda().eval().requires_grad_(False)
        self.backbone = build_backbone(self.config).cuda().eval().requires_grad_(False)
        self.text_cache = TextFeatureCache(self.backbone, torch.device('cuda'))
        self.builder = PlanningStateBuilder(data_root, CityReferCatalog(Path(data_root)/'cityrefer'),
            canvas_size=self.config['model']['canvas_size'], field_size=self.config['model']['field_size'],
            **self.config['state'])
        self.records = {}
        for split in ('val_seen','val_unseen'):
            for record in CityNavDataset(data_root, split):
                self.records[(split, record.episode_id)] = SimpleNamespace(
                    map_name=record.map_name, object_id=record.object_id, description_id=record.description_id,
                    episode_id=record.episode_id, instruction=record.instruction)
        self.metadata = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            epoch=payload['completed_epoch'], loaded_selector_tensors=len(weights), missing_disabled_spatial=missing,
            source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(source)/'sbfnav').rglob('*.py')})

    @torch.no_grad()
    def choose(self, *, episode_ids, instructions, histories, candidate_world_xy, split, active):
        # Only IDs, language, executed poses and predicted candidate coordinates enter this API.
        from sbfnav.candidate_proposal import Candidate
        from sbfnav.runtime import build_selector_inputs
        indices = np.flatnonzero(active)
        chosen = np.zeros(len(active), dtype=np.int64)
        if not len(indices):
            return chosen
        states, batches, texts = [], [], []
        for i in indices:
            record = self.records[(split, tuple(episode_ids[i]))]
            if instructions[i] != record.instruction:
                raise ValueError('HETT/SBF instruction mismatch')
            states.append(self.builder.build_from_poses(record, histories[i]))
            batches.append([Candidate(np.asarray(xy), ('hett_heatmap',)) for xy in candidate_world_xy[i]])
            texts.append(instructions[i])
        cfg = self.config['selector']
        inputs = build_selector_inputs(states, batches, self.text_cache.get(texts), self.backbone, self.text_cache,
            device=torch.device('cuda'), crop_extent_m=cfg['crop_extent_m'], crop_size=cfg['crop_size'],
            vision_batch_size=32, maximum_landmarks=cfg['max_landmarks'])
        scores = self.selector(inputs)
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite transferred selector logits')
        chosen[indices] = scores.argmax(-1).cpu().numpy()
        return chosen
