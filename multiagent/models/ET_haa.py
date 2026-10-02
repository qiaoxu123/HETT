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


def aggregate_history_grid(grid_fts, grid_indices, text_fts, grid_proj, cell_count):
    """Language-weight history features per grid cell without Python loops."""
    batch_size, history_len, feature_dim = grid_fts.shape
    if history_len == 0:
        return grid_fts.new_zeros((batch_size, cell_count, feature_dim))
    history = grid_fts.to(torch.float32)
    indices = grid_indices.to(dtype=torch.long)
    scores = torch.bmm(history, text_fts).max(dim=-1).values
    projected = grid_proj(history)
    cell_ids = torch.arange(cell_count, device=indices.device)
    cell_mask = indices.unsqueeze(-1) == cell_ids.view(1, 1, -1)
    has_history = cell_mask.any(dim=1, keepdim=True)
    masked_scores = scores.unsqueeze(-1).masked_fill(~cell_mask, -torch.inf)
    masked_scores = torch.where(has_history, masked_scores, torch.zeros_like(masked_scores))
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
        self.map_encoder = MapEncoder(240, input_channels=3)
        # Preserve baseline RNG so optional branches do not change any
        # original HETT parameter initialization under the same seed.
        _rng_state = torch.get_rng_state()
        self.global_landmark_encoder = MapEncoder(240, input_channels=1)
        torch.set_rng_state(_rng_state)
        self.global_landmark_gate = nn.Parameter(torch.tensor(0.0))
        self.encoder_vl = EncoderVL(args)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        self.centroid_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        _rng_state = torch.get_rng_state()
        self.landmark_anchor_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12),
        )
        torch.set_rng_state(_rng_state)
        self.landmark_anchor_type = nn.Parameter(torch.zeros(1, 1, self.args.demb))
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
        # Minimal SBF-style language conditioning for the heatmap only.
        # The zero-initialized scalar gate makes the initial forward path
        # identical to the existing heatmap branch.
        self.heatmap_candidate_norm = nn.LayerNorm(self.args.demb)
        self.heatmap_language_norm = nn.LayerNorm(self.args.demb)
        self.heatmap_lang_gate = nn.Parameter(torch.tensor(0.0))

        self.decoder_2_logits_full = nn.Sequential(
            nn.Linear(self.args.demb, self.args.demb // 2),
            nn.ReLU(),
            nn.Linear(self.args.demb // 2, 1),
        )
        # Optional trajectory geometry on the same 7x7 HETT candidates.
        # Spatial mode scores are shared with the heatmap head; this avoids the
        # redundant 7x7 -> 28x28 belief upsampling path.
        _rng_state = torch.get_rng_state()
        self.trajectory_geometry_head = nn.Sequential(
            nn.Linear(self.args.demb, self.args.demb // 2),
            nn.ReLU(),
            nn.Linear(
                self.args.demb // 2,
                2 + self.args.trajectory_steps * 2,
            ),
        )
        # Start from cell centers + straight fixed-horizon anchors.
        nn.init.zeros_(self.trajectory_geometry_head[-1].weight)
        nn.init.zeros_(self.trajectory_geometry_head[-1].bias)
        torch.set_rng_state(_rng_state)
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
        _rng_state = torch.get_rng_state()
        self.fc_global_map = nn.Linear(self.global_landmark_encoder.out_features, args.demb)
        torch.set_rng_state(_rng_state)

        self.text_proj = nn.Linear(768, 768)
        self.grid_proj = nn.Linear(768, 768)

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang = inputs["lang"]

        map_feat = self.map_encoder(inputs['maps'])
        global_map_feat = None
        if not self.args.disable_global_landmark_prior:
            global_map_feat = self.global_landmark_encoder(
                inputs['global_landmark_prior']
            )

        emb_candidates = self.candidate_encoder(inputs['candidates']) * emb_lang[:, :1, :]
        landmark_anchor_mask = inputs.get('landmark_anchor_mask')
        landmark_anchors = inputs.get('landmark_anchors')
        if landmark_anchors is None:
            emb_landmark_anchors = None
        else:
            emb_landmark_anchors = self.landmark_anchor_encoder(landmark_anchors) + self.landmark_anchor_type
            if landmark_anchor_mask is not None:
                emb_landmark_anchors = emb_landmark_anchors * landmark_anchor_mask.unsqueeze(-1).to(emb_landmark_anchors.dtype)
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
        if global_map_feat is not None:
            emb_global_maps = self.fc_global_map(global_map_feat).view(
                im_feature.shape[0], -1, 768
            )
            emb_maps = (
                emb_maps
                + torch.tanh(self.global_landmark_gate) * emb_global_maps
            )
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
            emb_landmark_anchors=emb_landmark_anchors,
            landmark_anchor_mask=landmark_anchor_mask,

            # inputs['lenths']
        )

        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1]]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 1]
        # encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3: emb_lang.shape[1] + 3 + emb_candidates.shape[1]]
        landmark_anchor_count = 0 if emb_landmark_anchors is None else emb_landmark_anchors.shape[1]
        encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3 + landmark_anchor_count:]
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

        # Explicit language-conditioned spatial belief:
        # each grid cell queries the instruction tokens before heatmap decoding.
        heatmap_query = self.heatmap_candidate_norm(target_decoder_input)
        heatmap_language = self.heatmap_language_norm(emb_lang)
        heatmap_attn_logits = torch.matmul(
            heatmap_query,
            heatmap_language.transpose(1, 2),
        ) / math.sqrt(self.args.demb)

        lang_mask = inputs.get("lang_mask")
        if lang_mask is not None:
            heatmap_attn_logits = heatmap_attn_logits.masked_fill(
                ~lang_mask[:, None, :].bool(),
                -torch.inf,
            )

        heatmap_attn = torch.softmax(heatmap_attn_logits, dim=-1)
        heatmap_lang_context = torch.matmul(heatmap_attn, heatmap_language)

        conditioned_target = (
            target_decoder_input
            + torch.tanh(self.heatmap_lang_gate) * heatmap_lang_context
        )

        # One logit per original 7x7 global grid cell.
        target_logits = self.decoder_2_logits_full(conditioned_target).squeeze(-1)

        trajectory_belief = None
        if self.args.enable_trajectory_belief:
            geometry = self.trajectory_geometry_head(conditioned_target)
            # Offset is constrained to half a 7x7 cell so each discrete mode
            # refines continuously inside its own spatial region.
            endpoint_offsets = (
                torch.tanh(geometry[:, :, :2])
                * (0.5 / float(self.args.grid_size))
            )
            residuals = geometry[:, :, 2:].reshape(
                batch_size,
                self.args.grid_size ** 2,
                self.args.trajectory_steps,
                2,
            )
            residuals = (
                torch.tanh(residuals) * self.args.trajectory_residual_scale
            )
            trajectory_belief = {
                # Reuse the heatmap logits as trajectory-mode probabilities.
                'logits': target_logits,
                'endpoint_offsets': endpoint_offsets,
                'residuals': residuals,
            }

        return (
            direction,
            progress,
            pred_goals,
            target_logits,
            trajectory_belief,
            emb_frames + emb_directions,
        )
