import torch
from .enc_visual import FeatureFlat
from .enc_vl import EncoderVL
from .encodings import DatasetLearnedEncoding
from . import model_util
from torch import nn
from torch.nn import functional as F

import numpy as np

from .goal_predictor import MapEncoder
from multiagent.two_stage import (
    MultiHypothesisConditioning,
    RelationalCandidateBeliefHead,
    SparseQuadtreeBeliefHead,
    TargetConditioning,
    select_action_coordinates,
)


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


class BidirectionalCrossAttention(nn.Module):
    """Exchange information between target and motion task tokens."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.target_from_motion = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.motion_from_target = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.target_attn_norm = nn.LayerNorm(d_model)
        self.motion_attn_norm = nn.LayerNorm(d_model)
        self.target_ffn_norm = nn.LayerNorm(d_model)
        self.motion_ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 与当前 EncoderVL 的 dim_feedforward 保持一致，避免引入过大的参数量。
        self.target_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.motion_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, target_tokens, motion_tokens):
        # 两个方向都从交互前的特征并行计算，避免某一方向先更新造成顺序偏置。
        target_delta, _ = self.target_from_motion(
            query=target_tokens,
            key=motion_tokens,
            value=motion_tokens,
            need_weights=False,
        )
        motion_delta, _ = self.motion_from_target(
            query=motion_tokens,
            key=target_tokens,
            value=target_tokens,
            need_weights=False,
        )

        target_tokens = self.target_attn_norm(target_tokens + self.dropout(target_delta))
        motion_tokens = self.motion_attn_norm(motion_tokens + self.dropout(motion_delta))

        target_tokens = self.target_ffn_norm(target_tokens + self.dropout(self.target_ffn(target_tokens)))
        motion_tokens = self.motion_ffn_norm(motion_tokens + self.dropout(self.motion_ffn(motion_tokens)))

        return target_tokens, motion_tokens


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        self.args = args
        # --------------- 0. 模型骨架：前端编码 + 融合编码 + 多头解码 -----------------
        # encoder and visual embeddings
        self.map_encoder = MapEncoder(240)
        self.encoder_vl = EncoderVL(args)
        self.candidate_encoder = nn.Sequential(      # 把候选点坐标 [B, N, 2] 映射到 embedding 空间 [B, N, d_model]
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        self.centroid_encoder = nn.Sequential(       # 把目标中心点 [B, N, 2] 映射到 embedding 空间 [B, N, d_model]
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        # # feature embeddings
        # self.vis_feat = FeatureFlat(input_shape=self.visual_tensor_shape, output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None

        # self.vis_feat = FeatureFlat(input_shape=(650,7,7), output_size=args.demb)

        self.args = args

        # --------------- 1. 多头解码器：action / progress / goal / target -----------------
        # action head: 输出 2 维方向向量，表示下一步动作方向
        # 输入 [B, d_model] -> 输出 [B, 2]
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

        self.attention_layer_vision = SoftDotAttention(49) # 让语言特征关注视觉帧；输入为 [B, dim] 与 [B, seq_len, dim]

        # progress head: 输出单值进度 score，表示任务推进程度
        self.decoder_2_progress_full = nn.Sequential(
            nn.Linear(self.args.demb, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.decoder_2_stop_full = nn.Sequential(
            nn.Linear(self.args.demb, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        # target logits head: 给每个候选位置打分，输出 [B, N, 1]
        self.decoder_2_logits_full = nn.Sequential(
            nn.Linear(self.args.demb, self.args.demb // 2),
            nn.ReLU(),
            nn.Linear(self.args.demb // 2, 1),
        )

        # goal head: 预测归一化目标位置 [x, y]，输出 [B, 2]
        self.decoder_2_goal_full = nn.Sequential(
            nn.Linear(self.args.demb, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

        # pose embedding: 把当前的 [sin(yaw), cos(yaw), x, y] 映射到 d_model 维
        self.direction_embedding = nn.Linear(4, self.args.demb)

        # 视觉帧展平：从 [B, T, 49] -> [B, T, d_model]
        self.fc2 = nn.Linear(49, self.args.demb)

        # 地图特征投影：把 map_encoder 的输出从 out_features 映射到 d_model
        self.fc_map = nn.Linear(self.map_encoder.out_features, args.demb)

        # 文本和 grid 特征投影，保证 language / historical grid 在同一向量空间内做相似度计算
        # 输入/输出均为 [B, seq_len, 768]
        self.text_proj = nn.Linear(768, 768)
        self.grid_proj = nn.Linear(768, 768)

        # target 侧包含 map/goal token 与所有 candidate tokens；motion 侧包含 action/progress tokens。
        self.task_interaction = BidirectionalCrossAttention(
            d_model=self.args.demb,
            num_heads=self.args.encoder_heads,
            dropout=self.args.dropout_transformer_encoder,
        )
        self.target_conditioning = TargetConditioning(
            d_model=self.args.demb,
            dropout=self.args.dropout_transformer_encoder,
        )
        candidate_heads = min(
            getattr(self.args, 'candidate_attention_heads', 4),
            self.args.encoder_heads,
        )
        while self.args.demb % candidate_heads != 0 and candidate_heads > 1:
            candidate_heads -= 1
        self.candidate_belief = RelationalCandidateBeliefHead(
            d_model=self.args.demb,
            num_heads=candidate_heads,
            hidden_dim=getattr(self.args, 'candidate_hidden_dim', 256),
            topk=getattr(self.args, 'candidate_topk', 4),
            max_offset=0.5 / max(getattr(self.args, 'candidate_grid_size', 8), 1),
            map_meters=getattr(self.args, 'map_meters', 410.0),
            local_gate_center_m=getattr(
                self.args, 'candidate_local_gate_center_m', 100.0
            ),
            local_gate_temperature_m=getattr(
                self.args, 'candidate_local_gate_temperature_m', 30.0
            ),
            dropout=self.args.dropout_transformer_encoder,
        )
        self.multi_target_conditioning = MultiHypothesisConditioning(
            d_model=self.args.demb,
            num_heads=candidate_heads,
            dropout=self.args.dropout_transformer_encoder,
        )
        self.quadtree_belief = SparseQuadtreeBeliefHead(
            d_model=self.args.demb,
            depth=getattr(self.args, 'quadtree_depth', 5),
            topk=getattr(self.args, 'quadtree_topk', 4),
            hidden_dim=getattr(self.args, 'quadtree_hidden_dim', 256),
        )
        self.last_quadtree_loss = None
        self.last_quadtree_leaf_centers = None
        self.last_quadtree_leaf_probs = None
        self.last_candidate_logits = None
        self.last_refined_candidates = None
        self.last_topk_coordinates = None
        self.last_topk_probs = None
        self.last_topk_indices = None
        self.last_candidate_global_logits = None
        self.last_candidate_local_logits = None
        self.last_candidate_local_gate = None

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # --------------- 2. 输入整理：语言 / 地图 / 视觉 / 方向 / 候选 -----------------
        # emb_lang:             [B, L_lang, d_model]
        # inputs['maps']:       [B, H_map, W_map, C_map] or [B, C_map, H_map, W_map]
        # inputs['candidates']: [B, N_cand, 2]
        # inputs['directions']: [B, T_dir, 4]  # sin(yaw), cos(yaw), x, y
        # inputs['frames']:     [B, T_frame, 512, 49]
        output = {}
        emb_lang = inputs["lang"]  # [B, L_lang, d_model]
        map_feat = self.map_encoder(inputs['maps']) # [B, C, H', W']

        # Global branch starts from coordinate-only candidates. It is kept
        # separate from the multimodal/local branch so far-away grounding can
        # rely on language + global geometry instead of noisy current RGB.
        global_candidate_tokens = self.candidate_encoder(
            inputs['candidates']
        )
        # Legacy/local branch keeps the released language modulation and later
        # receives history/map/vision information through EncoderVL.
        emb_candidates = global_candidate_tokens * emb_lang[:, :1, :]
        # print(torch.isnan(map_feat).any(), torch.isinf(map_feat).any())

        # --------------- 3. 视觉帧注意力：语言关注每一帧 -----------------
        im_feature = inputs["frames"]  # [B, T_frame, 512, 49]
        att_frame_feature = torch.zeros((im_feature.shape[0], 0, 49)).cuda() # [B, T_frame, 49]
        for i in range(im_feature.shape[1]):
            att_single_frame_feature, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:, i, :, :]) # [B, 49], [B, 49]
            att_frame_feature = torch.concat((att_frame_feature, att_single_frame_feature.unsqueeze(1)), axis=1) # [B, T_frame, 49]

        emb_frames = self.fc2(att_frame_feature.view(-1, 49)).view(*im_feature.shape[:2], -1) # [B, T_frame, d_model]

        # --------------- 4. 地图特征与方向特征编码 -----------------
        emb_maps = self.fc_map(map_feat).view(im_feature.shape[0], -1, 768) # [B, N_map, d_model]

        emb_directions = self.direction_embedding(  # [B, T_dir, d_model]
            inputs["directions"].view(-1, 4)).view( # [B, T_dir, 4], [sin(yaw), cos(yaw), x, y]
            im_feature.shape[0], -1, 768
        )
        batch_size = emb_lang.shape[0]

        # --------------- 5. 历史 grid map：将过去的空间记忆压成 5x5 结构化 token -----------------
        grid_map_input = torch.zeros(batch_size, self.args.grid_size ** 2, 768).cuda() # [B, grid_size^2, 768]

        text_fts = self.text_proj(emb_lang).permute(0, 2, 1) # [B, 768, L_lang]，用于和历史网格特征做相似度计算
        grid_masks = [[] for b in range(batch_size)]
        max_cell_num = self.args.grid_size ** 2
        grid_fts = inputs['grid_fts']               # 历史网格特征，通常是 [B, N_hist, 768]
        grid_map_indexs = inputs['grid_index']      # 每个历史特征对应到哪个 cell，通常是 [B, N_hist]
        for b in range(batch_size):
            # tmp_fts: [N_hist, 768]
            tmp_fts = grid_fts[b].to(torch.float32)
            
            # 通过和语言特征的相似度，计算每个历史 cell 与当前指令的相关性
            relevance = tmp_fts @ text_fts[b]
            if inputs.get('lang_mask') is not None:
                relevance = relevance.masked_fill(~inputs['lang_mask'][b].bool(), -float('inf'))
            grid_fts_weight, _ = relevance.max(dim=-1) # [N_hist]
            tmp_fts = self.grid_proj(tmp_fts)

            for i in range(self.args.grid_size ** 2):
                # cell_fts: [num_hist_in_cell, 768]
                cell_fts = tmp_fts[grid_map_indexs[b] == i]
                if cell_fts.shape[0] == 0:
                    grid_masks[b].append(0)
                else:
                    grid_masks[b].append(1)

                # 对同一 cell 中的历史特征按语言相关性做加权求和，得到该 cell 的汇总表示
                # 结果：grid_map_input[b, i] -> [768]
                grid_map_input[b, i] = (
                    cell_fts * torch.softmax(grid_fts_weight[grid_map_indexs[b] == i], dim=-1).unsqueeze(-1)
                ).sum(-2)

            # if max_cell_num < sum(grid_masks[b]):
            #     max_cell_num = sum(grid_masks[b])
        # grid_masks = torch.tensor(grid_masks).cuda()
        grid_map_embeds = torch.zeros(batch_size, max_cell_num, 768).to(grid_fts[0].device) # [B, max_cell_num, 768]

        # 将历史空间记忆对齐到任意候选集。候选可以是 5x5 旧网格，也可以是
        # 8x8 sparse candidate grid；每个候选读取其所在 historical cell 的记忆。
        candidate_xy = inputs['candidates'].clamp(0.0, 1.0 - 1e-6)
        hist_xy = torch.floor(candidate_xy * self.args.grid_size).long()
        hist_ids = hist_xy[..., 0] * self.args.grid_size + hist_xy[..., 1]
        candidate_history = grid_map_input.gather(
            1, hist_ids.unsqueeze(-1).expand(-1, -1, grid_map_input.shape[-1])
        )
        emb_candidates = emb_candidates + candidate_history

        # --------------- 6. Transformer 融合：把所有模态拼接后做跨模态 self-attention -----------------
        encoder_out, _ = self.encoder_vl.forward_with_map(
            emb_lang,           # [B, L_lang, d]
            emb_frames,         # [B, T_frame, d]
            emb_directions,     # [B, T_dir, d]
            emb_maps,           # [B, N_map, d]
            emb_candidates,     # [B, N_cand, d]
            lang_mask=inputs.get('lang_mask'),
        )

        # --------------- 7. 按真实 token 长度切分 Transformer 输出 -----------------
        lang_end = emb_lang.shape[1]
        frame_end = lang_end + emb_frames.shape[1]
        direction_end = frame_end + emb_directions.shape[1]
        map_end = direction_end + emb_maps.shape[1]

        encoder_out_frames = encoder_out[:, lang_end:frame_end]
        encoder_out_directions = encoder_out[:, frame_end:direction_end]
        encoder_out_maps = encoder_out[:, direction_end:map_end]
        encoder_out_candidates = encoder_out[:, map_end:]

        # 当前 rollout 只输入一个 frame/direction/map token；取最后一个也兼容后续历史扩展。
        encoder_out_visual = encoder_out_frames[:, -1]          # [B, d_model]
        encoder_out_direction = encoder_out_directions[:, -1]   # [B, d_model]
        encoder_out_map = encoder_out_maps[:, -1]               # [B, d_model]

        # --------------- 8. target 与 action/progress 双向交互 -----------------
        target_tokens = torch.cat((encoder_out_map.unsqueeze(1), encoder_out_candidates), dim=1)  # [B, 1 + N_cand, d_model]
        motion_tokens = torch.stack((encoder_out_direction, encoder_out_visual), dim=1)           # [B, 2, d_model]
        training_stage = getattr(self.args, 'training_stage', 'joint')
        if (training_stage == 'joint'
                and not getattr(self.args, 'disable_task_interaction', False)):
            target_tokens, motion_tokens = self.task_interaction(target_tokens, motion_tokens)

        # --------------- 9. 多头输出：direction / progress / goal / target -----------------
        goal_decoder_input = target_tokens[:, 0]               # [B, d_model]
        target_decoder_input = target_tokens[:, 1:]            # [B, N_cand, d_model]
        pred_goals = self.decoder_2_goal_full(goal_decoder_input) # [B, 2] 归一化目标位置
        self.last_quadtree_loss = None
        self.last_quadtree_leaf_centers = None
        self.last_quadtree_leaf_probs = None
        self.last_candidate_logits = None
        self.last_refined_candidates = None
        self.last_topk_coordinates = None
        self.last_topk_probs = None
        self.last_topk_indices = None

        if getattr(self.args, 'target_representation', 'point') == 'candidates':
            current_positions = inputs['directions'][:, -1, 2:4]
            candidate_belief = self.candidate_belief(
                target_decoder_input,
                inputs['candidates'],
                emb_lang,
                current_positions,
                language_mask=inputs.get('lang_mask'),
                global_candidate_tokens=global_candidate_tokens,
            )
            pred_goals = candidate_belief.coordinate
            self.last_candidate_logits = candidate_belief.logits
            self.last_refined_candidates = candidate_belief.refined_coordinates
            self.last_topk_coordinates = candidate_belief.topk_coordinates
            self.last_topk_probs = candidate_belief.topk_probs
            self.last_topk_indices = candidate_belief.topk_indices
            self.last_candidate_global_logits = candidate_belief.global_logits
            self.last_candidate_local_logits = candidate_belief.local_logits
            self.last_candidate_local_gate = candidate_belief.local_gate
        if (training_stage == 'target'
                and getattr(self.args, 'target_representation', 'point') == 'quadtree'):
            quadtree_truth = inputs.get('target_coordinates')
            quadtree_context = goal_decoder_input
            active_mask = inputs.get('target_active_mask')
            if quadtree_truth is not None and active_mask is not None:
                active_mask = active_mask.bool()
                if active_mask.any():
                    hierarchy_loss = self.quadtree_belief.hierarchy_loss(
                        quadtree_context[active_mask],
                        quadtree_truth[active_mask],
                    )
                else:
                    hierarchy_loss = quadtree_context.new_zeros(())
                quadtree = self.quadtree_belief(
                    quadtree_context,
                    truth=None,
                )
                quadtree.hierarchy_loss = hierarchy_loss
            else:
                quadtree = self.quadtree_belief(
                    quadtree_context,
                    truth=quadtree_truth,
                )
            pred_goals = quadtree.coordinate
            self.last_quadtree_loss = quadtree.hierarchy_loss
            self.last_quadtree_leaf_centers = quadtree.leaf_centers
            self.last_quadtree_leaf_probs = quadtree.leaf_probs
        if training_stage == 'action':
            current_positions = inputs['directions'][:, -1, 2:4]
            if (getattr(self.args, 'target_representation', 'point') == 'candidates'
                    and self.last_topk_coordinates is not None):
                target_coordinates = self.last_topk_coordinates.detach()
                target_probs = self.last_topk_probs.detach()
                truth = inputs.get('target_coordinates')
                use_truth = inputs.get('target_coordinate_mask')
                if truth is not None and use_truth is not None and use_truth.any():
                    mask = use_truth.bool()
                    target_coordinates = target_coordinates.clone()
                    target_probs = target_probs.clone()
                    target_coordinates[mask, 0] = truth.detach()[mask]
                    if target_probs.shape[1] == 1:
                        target_probs[mask, 0] = 1.0
                    else:
                        remaining = 0.3 / (target_probs.shape[1] - 1)
                        target_probs[mask] = remaining
                        target_probs[mask, 0] = 0.7
                motion_tokens = self.multi_target_conditioning(
                    motion_tokens,
                    target_coordinates,
                    target_probs,
                    current_positions,
                )
            else:
                action_targets = select_action_coordinates(
                    predicted=pred_goals,
                    truth=inputs.get('target_coordinates'),
                    use_truth=inputs.get('target_coordinate_mask'),
                )
                motion_tokens = self.target_conditioning(
                    motion_tokens,
                    action_targets,
                    current_positions,
                )

        action_decoder_input = motion_tokens[:, 0]             # [B, d_model]
        decoder_input = motion_tokens[:, 1]                    # [B, d_model]

        output = self.decoder_2_action_full(action_decoder_input) # [B, 2] 归一化方向向量
        norm = torch.norm(output, dim=1, keepdim=True) + 1e-6     # 避免除零
        direction = output / norm

        # progress or stop probability: [B, 1]
        if training_stage == 'action' and getattr(self.args, 'use_stop_head', False):
            progress = torch.sigmoid(self.decoder_2_stop_full(decoder_input))
        else:
            progress = self.decoder_2_progress_full(decoder_input)

        # target_logits: [B, N_cand, 1]
        target_logits = self.decoder_2_logits_full(target_decoder_input)

        return direction, progress, pred_goals, target_logits, emb_frames + emb_directions
