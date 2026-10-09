import math
import torch
from .enc_visual import FeatureFlat
from .enc_vl import EncoderVL
from .encodings import DatasetLearnedEncoding
from . import model_util
from torch import nn
from torch.nn import functional as F

import numpy as np

from .goal_predictor import MapEncoder
from .spatial_belief import CompactSpatialBelief
from .relative_geometry import dense_relative_geometry
from .heatmap_trajectory import HeatmapTrajectoryHead


def aggregate_history_grid(grid_fts, grid_indices, text_fts, grid_proj, cell_count):
    """Language-weight history features per grid cell without Python loops."""
    batch_size, history_len, feature_dim = grid_fts.shape
    if history_len == 0:
        return grid_fts.new_zeros((batch_size, cell_count, feature_dim))

    history = grid_fts.to(torch.float32)
    indices = grid_indices.to(dtype=torch.long)
    scores = torch.bmm(history, text_fts).amax(dim=-1)
    projected = grid_proj(history)

    cell_ids = torch.arange(cell_count, device=indices.device)
    cell_mask = indices.unsqueeze(-1) == cell_ids.view(1, 1, -1)
    has_history = cell_mask.any(dim=1, keepdim=True)
    masked_scores = scores.unsqueeze(-1).masked_fill(~cell_mask, -torch.inf)
    masked_scores = torch.where(
        has_history, masked_scores, torch.zeros_like(masked_scores)
    )
    weights = torch.softmax(masked_scores, dim=1) * cell_mask
    return torch.einsum('btc,btd->bcd', weights, projected)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

        # self.c = nn.Sequential(
        # nn.Linear(768, 256),
        # # nn.BatchNorm1d(64, eps=1e-12),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(256, 32),
        # # nn.BatchNorm1d(64, eps=1e-12),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(32, 4),
        # # nn.BatchNorm1d(768, eps=1e-12),
        # nn.ReLU())

    def forward(self, h, context, mask=None):  # context will be weighted and concat with h
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        lang_embeds = torch.cat((weighted_context, h), 1)

        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        self.args = args
        # encoder and visual embeddings
        self.map_encoder = MapEncoder(240)
        self.encoder_vl = EncoderVL(args)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        self.centroid_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        # # feature embeddings
        # self.vis_feat = FeatureFlat(input_shape=self.visual_tensor_shape, output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None

        # self.vis_feat = FeatureFlat(input_shape=(650,7,7), output_size=args.demb)

        self.args = args

        # XVIEW
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(self.args.demb, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        self.attention_layer_vision = SoftDotAttention(49)
        self.decoder_2_progress_full = nn.Sequential(
            nn.Linear(self.args.demb, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        # Dedicated dense belief field. It consumes the full 4-channel map
        # [current, explored, global landmarks, referenced landmarks] and
        # produces a language-conditioned 28x28 spatial belief.
        self.spatial_belief = CompactSpatialBelief(
            input_channels=4,
            field_size=self.args.heatmap_grid_size,
            hidden_dim=256,
            language_dim=self.args.demb,
            attention_heads=8,
            dropout=0.1,
        )
        self.trajectory_head = HeatmapTrajectoryHead(
            feature_dim=256,
            language_dim=args.demb,
            modes=getattr(args, 'trajectory_modes', 3),
            waypoints=getattr(args, 'trajectory_waypoints', 8),
        )
        if getattr(args, 'trajectory_relation_observation', False):
            from .relation_observation import RelationObservation
            self.trajectory_head.relation_observation = RelationObservation(language_dim=args.demb)
        self.decoder_2_goal_full = nn.Sequential(
            nn.Linear(self.args.demb, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
        self.direction_embedding = nn.Linear(4, self.args.demb)

        self.fc2 = nn.Linear(49, self.args.demb)

        self.fc_map = nn.Linear(self.map_encoder.out_features, args.demb)

        self.text_proj = nn.Linear(768, 768)
        self.grid_proj = nn.Linear(768, 768)

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang = inputs["lang"]

        maps = inputs['maps']
        if maps.shape[1] == 4:
            # Preserve the legacy ET map path exactly: current view, explored,
            # referenced landmarks. The new global-landmark channel is used
            # only by the dense belief branch.
            legacy_maps = torch.cat((maps[:, :2], maps[:, 3:4]), dim=1)
        elif maps.shape[1] == 3:
            legacy_maps = maps
        else:
            raise ValueError(f"expected 3 or 4 map channels, got {maps.shape[1]}")
        map_feat = self.map_encoder(legacy_maps)

        emb_candidates = self.candidate_encoder(inputs['candidates']) * emb_lang[:, :1, :]
        # print(torch.isnan(map_feat).any(), torch.isinf(map_feat).any())

        # # embed frames and direiction (650,49) --> 768
        # im_feature = inputs["frames"]
        # embed_frame, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:,-1, :, :])
        # h_sali = self.fc(embed_frame).view(-1,1,8,8)
        # pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # frames_pad_emb = self.vis_feat(im_feature.view(-1, 650,7,7)).view(*im_feature.shape[:2], -1)

        # embed frames and direiction (1,49) --> 768
        im_feature = inputs["frames"]
        if im_feature.shape[1] == 1:
            att_single_frame_feature, beta = self.attention_layer_vision(
                inputs["lang_cls"], im_feature[:, 0, :, :]
            )
            att_frame_feature = att_single_frame_feature.unsqueeze(1)
        else:
            attended = [
                self.attention_layer_vision(
                    inputs["lang_cls"], im_feature[:, i, :, :]
                )[0]
                for i in range(im_feature.shape[1])
            ]
            att_frame_feature = torch.stack(attended, dim=1)

        emb_frames = self.fc2(att_frame_feature.view(-1, 49)).view(*im_feature.shape[:2], -1)

        emb_maps = self.fc_map(map_feat).view(im_feature.shape[0], -1, 768)
        # print('sss', emb_frames.shape, emb_maps.shape)
        # print(map_feat.shape)

        emb_directions = self.direction_embedding(inputs["directions"].view(-1, 4)).view(im_feature.shape[0], -1,
                                                                                         768)  # (batch, embedding_size)
        batch_size = emb_lang.shape[0]

        text_fts = self.text_proj(emb_lang).permute(0, 2, 1)
        max_cell_num = self.args.grid_size ** 2
        grid_fts = inputs['grid_fts']
        grid_map_indexs = inputs['grid_index']
        grid_map_input = aggregate_history_grid(
            grid_fts,
            grid_map_indexs,
            text_fts,
            self.grid_proj,
            max_cell_num,
        )

        emb_candidates = emb_candidates + grid_map_input

        # emb_centroids = (self.centroid_encoder(inputs['centroids']) * emb_lang[:, 0, :]).view(im_feature.shape[0], -1, 768)
        # emb_centroids = self.centroid_encoder(inputs['centroids']).view(im_feature.shape[0], -1, 768)
        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl.forward_with_map(
            emb_lang,
            emb_frames,
            emb_directions,
            emb_maps,
            emb_candidates,

            # inputs['lenths']
        )

        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1]]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 1]
        # encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3: emb_lang.shape[1] + 3 + emb_candidates.shape[1]]
        encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3:]
        encoder_out_centroids = encoder_out[:, emb_lang.shape[1] + 2]
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_decoder_input = encoder_out_direction.reshape(-1, self.args.demb)
        goal_decoder_input = encoder_out_centroids.reshape(-1, self.args.demb)
        target_decoder_input = encoder_out_candidates.reshape(-1, max_cell_num, self.args.demb)

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        # goal_logits = self.decoder_2_goal_full(goal_decoder_input)
        pred_goals = self.decoder_2_goal_full(goal_decoder_input)
        norm = torch.norm(output, dim=1, keepdim=True) + 1e-6  # 避免除以零
        direction = output / norm

        progress = self.decoder_2_progress_full(decoder_input)

        # Dense language-conditioned spatial belief. This path is independent
        # of the legacy 7x7 candidate/history tokens used by the controller.
        geometry = None
        if getattr(self.args, 'heatmap_relative_geometry', True):
            if 'directions' not in inputs:
                raise ValueError('relative geometry requires current pose and yaw')
            current_pose = inputs['directions'][:, -1, :]
            if current_pose.shape[-1] != 4:
                raise ValueError('direction tokens must contain sin/cos yaw and normalized xy')
            # Landmark centroids from the instruction-referenced map channel.
            reference_mask = F.adaptive_avg_pool2d(
                maps[:, 3:4], (self.args.heatmap_grid_size, self.args.heatmap_grid_size)
            ).squeeze(1).clamp_min(0)
            coord = (torch.arange(self.args.heatmap_grid_size, device=maps.device, dtype=maps.dtype) + 0.5) / self.args.heatmap_grid_size
            ref_mass = reference_mask.sum(dim=(1, 2))
            ref_x = (reference_mask * coord.view(1, 1, -1)).sum(dim=(1, 2)) / ref_mass.clamp_min(1e-6)
            ref_y = (reference_mask * coord.view(1, -1, 1)).sum(dim=(1, 2)) / ref_mass.clamp_min(1e-6)
            ref_centroid = torch.stack((ref_x, ref_y), dim=-1)
            # In the multi-landmark experiment, do not fuse all reference
            # contours into a single anonymous center. Keep the UAV geometry,
            # and delegate landmark-specific geometry to the relation head.
            if getattr(self.args, 'heatmap_multi_landmark', False):
                ref_centroid = torch.zeros_like(ref_centroid)
                reference_present = torch.zeros_like(ref_mass, dtype=torch.bool)
            else:
                reference_present = ref_mass > 1e-6
            geometry = dense_relative_geometry(
                current_pose[:, 2:4], current_pose[:, :2],
                ref_centroid, reference_present,
                field_size=self.args.heatmap_grid_size,
                map_meters=self.args.map_meters,
            )
        multi_inputs = {}
        if getattr(self.args, 'heatmap_multi_landmark', False):
            # Missing keys must fail loudly: silently falling back to the
            # aggregate mask would invalidate the multi-landmark experiment.
            names = ('landmark_xy', 'landmark_extent', 'landmark_valid',
                     'landmark_text_mask')
            for name in names:
                if name not in inputs:
                    raise ValueError(f"multi-landmark mode requires {name}")
                multi_inputs[name] = inputs[name]
        belief = self.spatial_belief(
            maps,
            emb_lang,
            inputs.get("lang_mask"),
            geometry=geometry,
            **multi_inputs,
        )
        target_logits = belief.logits.flatten(1)
        relation_state = None
        if getattr(self.args, 'trajectory_relation_observation', False):
            pose = inputs['directions'][:, -1]
            relation_state = self.trajectory_head.relation_observation(
                belief.spatial_features.detach(), pose[:, 2:4], pose[:, :2],
                inputs['relation_language_tokens'], inputs['lang_mask'],
                inputs['frames'][:, -1].mean(-1).detach(),
                inputs['landmark_xy'], inputs['landmark_extent'],
                inputs['landmark_valid'], inputs['landmark_text_mask'],
                inputs.get('trajectory_history_xy'))
            target_logits = target_logits + relation_state[0].flatten(1)

        if getattr(self.args, 'heatmap_trajectory_enabled', False):
            pose = inputs['directions'][:, -1]
            # Compact ranking only learns its own shared head; the ranking
            # loss must not backpropagate through the upstream heatmap.
            compact_mode = getattr(self.args, 'trajectory_compact_mode', False)
            proposal_features = belief.spatial_features.detach() if compact_mode else belief.spatial_features
            proposal_probs = belief.probabilities.detach() if compact_mode else belief.probabilities
            if relation_state is not None:
                proposal_features = belief.spatial_features.detach() + relation_state[1]
                proposal_probs = torch.softmax(target_logits, -1).reshape_as(belief.probabilities)
            proposals = self.trajectory_head(
                proposal_features, proposal_probs,
                pose[:, 2:4], pose[:, :2],
                top_k=getattr(self.args, 'trajectory_goal_k', 5),
                nms_kernel=self.args.heatmap_nms_kernel,
                teacher_goal=(None if getattr(self.args, 'trajectory_compact_mode', False)
                              else inputs.get('trajectory_teacher_goal')),
                language_tokens=emb_lang,
                language_mask=inputs.get('lang_mask'),
                landmark_xy=inputs.get('landmark_xy'),
                landmark_extent=inputs.get('landmark_extent'),
                landmark_valid=inputs.get('landmark_valid'),
                landmark_text_mask=inputs.get('landmark_text_mask'),
                history_xy=inputs.get('trajectory_history_xy'),
                relation_enabled=getattr(self.args, 'trajectory_relation_selector', True),
                selector_mode=getattr(self.args, 'trajectory_selector_mode', 'prior'),
                compact=getattr(self.args, 'trajectory_compact_mode', False),
                # Same step-length setting used by ALL waypoint variants.
                local_step_m=self.args.heatmap_waypoint_step_m,
                map_meters=self.args.map_meters,
            )
            generated, supervision = proposals
            if relation_state is not None:
                rows = torch.arange(pose.shape[0], device=pose.device)
                if getattr(self.args, 'trajectory_selector_mode', 'joint') == 'prior':
                    selected = proposal_probs.flatten(1).gather(1, generated.goal_ids).argmax(-1)
                else:
                    selected = generated.joint_logits.logsumexp(-1).argmax(-1)
                cell_ids = generated.goal_ids[rows, selected]
                generated.stop_logits = self.trajectory_head.relation_observation.stop(
                    relation_state[2], relation_state[3], cell_ids, proposal_probs)
            # Return plain nested tensors for torch DDP graph discovery.
            # Custom dataclass outputs are not reliably traversed by all
            # find_unused_parameters versions in distributed training.
            tensors = (generated.trajectories, generated.mode_logits,
                       generated.joint_logits, generated.goal_xy,
                       generated.goal_ids, generated.stop_logits,
                       generated.candidate_logits)
            return (direction, progress, pred_goals, target_logits,
                    emb_frames + emb_directions, (tensors, supervision))

        return direction, progress, pred_goals, target_logits, emb_frames + emb_directions
