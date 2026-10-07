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
from .spatial_belief import CompactSpatialBelief, greedy_nms_topk
from .candidate_selector import (
    CandidateVisualSelector,
    heatmap_ids_to_normalized_xy,
    sample_candidate_features_from_current_view,
)


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

        # Optional Top-K discriminator. It is intentionally independent of the
        # heatmap score and referenced-landmark map. The only candidate-specific
        # evidence is what is actually visible in the current RGB feature map,
        # conditioned on the instruction tokens.
        self.candidate_visual_selector = None
        if getattr(self.args, "candidate_selector", False):
            self.candidate_visual_selector = CandidateVisualSelector(
                visual_dim=512,
                language_dim=self.args.demb,
                hidden_dim=self.args.candidate_selector_hidden_dim,
                layers=self.args.candidate_selector_layers,
                attention_heads=self.args.candidate_selector_heads,
                dropout=self.args.candidate_selector_dropout,
                use_geometry=not self.args.candidate_selector_no_geometry,
                use_landmark_text=not self.args.candidate_selector_no_landmark_text,
            )

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
        belief = self.spatial_belief(
            maps,
            emb_lang,
            inputs.get("lang_mask"),
        )
        target_logits = belief.logits.flatten(1)

        selector_candidate_ids = None
        selector_logits = None
        selector_visible = None
        selector_candidate_xy = None
        selector_current_visual = None
        selector_current_visible = None
        if self.candidate_visual_selector is not None:
            selector_k = min(
                int(self.args.candidate_selector_top_k),
                int(self.args.heatmap_top_k),
            )
            selector_candidate_ids = greedy_nms_topk(
                belief.probabilities,
                top_k=selector_k,
                kernel_size=self.args.heatmap_nms_kernel,
            )
            selector_candidate_xy = heatmap_ids_to_normalized_xy(
                selector_candidate_ids,
                field_size=self.args.heatmap_grid_size,
            )

            current_direction = inputs["directions"][:, -1]
            if "view_radius_m" not in inputs:
                raise ValueError(
                    "view_radius_m is required when candidate_selector is enabled"
                )
            current_frame_map = im_feature[:, -1]
            selector_current_visual, selector_current_visible = (
                sample_candidate_features_from_current_view(
                    current_frame_map,
                    selector_candidate_xy,
                    current_direction[:, 2:4],
                    current_direction[:, 0],
                    current_direction[:, 1],
                    inputs["view_radius_m"],
                    map_meters=self.args.map_meters,
                )
            )

            if (
                "candidate_visual_memory" not in inputs
                or "candidate_visual_count" not in inputs
            ):
                raise ValueError(
                    "candidate visual memory is required when selector is enabled"
                )
            memory = inputs["candidate_visual_memory"]
            memory_count = inputs["candidate_visual_count"]
            expected_cells = self.args.heatmap_grid_size ** 2
            if memory.shape != (batch_size, expected_cells, 512):
                raise ValueError("candidate_visual_memory shape mismatch")
            if memory_count.shape != (batch_size, expected_cells):
                raise ValueError("candidate_visual_count shape mismatch")

            gather_index = selector_candidate_ids.unsqueeze(-1).expand(-1, -1, 512)
            historical_sum = memory.gather(1, gather_index)
            historical_count = memory_count.gather(1, selector_candidate_ids)
            evidence_sum = historical_sum + (
                selector_current_visual
                * selector_current_visible.unsqueeze(-1).to(
                    selector_current_visual.dtype
                )
            )
            evidence_count = (
                historical_count
                + selector_current_visible.to(historical_count.dtype)
            )
            selector_visible = evidence_count > 0
            candidate_visual = evidence_sum / evidence_count.clamp_min(
                1.0
            ).unsqueeze(-1)

            for required in (
                "selector_landmark_features",
                "selector_landmark_xy",
                "selector_landmark_mask",
            ):
                if required not in inputs:
                    raise ValueError(
                        f"{required} is required when candidate_selector is enabled"
                    )
            selector_logits = self.candidate_visual_selector(
                candidate_visual,
                emb_lang,
                selector_candidate_xy,
                current_direction[:, 2:4],
                inputs["selector_landmark_features"],
                inputs["selector_landmark_xy"],
                inputs["selector_landmark_mask"],
                inputs.get("lang_mask"),
            )

        # Candidate selector outputs are diagnostics/selection evidence only.
        # The heatmap logits remain unchanged and the selector never receives
        # heatmap probabilities or landmark-mask values.
        return (
            direction,
            progress,
            pred_goals,
            target_logits,
            emb_frames + emb_directions,
            selector_candidate_ids,
            selector_logits,
            selector_visible,
            selector_candidate_xy,
            selector_current_visual,
            selector_current_visible,
        )
