import torch

from . import model_util
from .encodings import PosEncoding, PosLearnedEncoding, TokenLearnedEncoding, MapPosEncoding, CanPosEncoding
# import model_util
from torch import nn
import numpy as np


class EncoderVL(nn.Module):
    def __init__(self, args):
        """
        transformer encoder for language, frames and action inputs
        """
        super(EncoderVL, self).__init__()

        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb,
            args.encoder_heads,
            args.demb,
            args.dropout_transformer_encoder,
        )
        self.enc_transformer = nn.TransformerEncoder(encoder_layer, args.encoder_layers)

        # how many last actions to attend to
        self.num_input_actions = args.num_input_actions

        # encodings
        self.enc_pos = PosEncoding(args.demb)
        self.enc_pos_map = MapPosEncoding(args.demb)
        self.enc_pos_can = CanPosEncoding(args.demb)
        self.enc_pos_learn = None
        self.enc_token = None
        self.enc_layernorm = nn.LayerNorm(args.demb)
        self.enc_dropout = nn.Dropout(args.dropout_emb, inplace=True)

    def forward(
            self,
            emb_lang,
            emb_frames,
            emb_directions,
            emb_positions,
            lengths
    ):
        """
        pass embedded inputs through embeddings and encode them using a transformer
        """
        length_max = np.max(lengths)
        # emb_lang is processed on each GPU separately so they size can vary
        length_lang = emb_lang.shape[1]
        # create a mask for padded elements
        length_mask_pad = length_lang + 2
        mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        # for i, l in enumerate(lengths):
        #     # mask padded frames
        #     mask_pad[i, (length_lang + l):(length_lang + length_max)] = True
        #     # mask padded directions
        #     mask_pad[i, (length_lang + length_max + l):] = True

        # encode the inputs
        emb_all = self.encode_inputs(emb_lang, emb_frames, emb_directions, emb_positions, length_lang, mask_pad)

        # create a mask for attention (prediction at t should not see frames at >= t+1)
        mask_attn = model_util.generate_attention_mask(
            length_lang,
            length_max,
            emb_all.device,
        )

        # print(emb_all.shape, mask_attn.shape, mask_pad.shape)

        # encode the inputs
        output = self.enc_transformer(emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
        return output, mask_pad

        # length_mask_pad = length_lang + length_max * 2
        # mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        # for i, l in enumerate(lengths):
        #     # mask padded frames
        #     mask_pad[i, (length_lang + l):(length_lang + length_max)] = True
        #     # mask padded directions
        #     mask_pad[i, (length_lang + length_max + l):] = True
        #
        # # encode the inputs
        # emb_all = self.encode_inputs(emb_lang, emb_frames, emb_directions, length_lang, mask_pad)
        #
        # # create a mask for attention (prediction at t should not see frames at >= t+1)
        # mask_attn = model_util.generate_attention_mask(
        #     length_lang,
        #     length_max,
        #     emb_all.device,
        # )
        #
        # print(emb_all.shape, mask_attn.shape, mask_pad.shape)
        #
        # # encode the inputs
        # output = self.enc_transformer(emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
        # return output, mask_pad


    def forward_with_map(
            self,
            emb_lang,
            emb_frames,
            emb_directions,
            emb_maps,
            emb_positions,
            emb_landmark_anchors=None,
            landmark_anchor_mask=None,
            # lengths
    ):
        """
        Encode HETT inputs, optionally adding instruction-referenced landmark
        centroids as padded geographic anchor tokens.

        The original grid candidates remain the final tokens in the sequence,
        so downstream grid-logit indexing and the two-stage controller are
        unchanged.
        """
        length_positions = emb_positions.shape[1]
        length_lang = emb_lang.shape[1]
        length_anchors = (
            0 if emb_landmark_anchors is None
            else emb_landmark_anchors.shape[1]
        )

        emb_lang, emb_frames, emb_directions, emb_maps, emb_positions = (
            self.enc_pos_map(
                emb_lang,
                emb_frames,
                emb_directions,
                emb_maps,
                emb_positions,
                length_lang,
            )
        )

        parts = [emb_lang, emb_frames, emb_directions, emb_maps]
        if emb_landmark_anchors is not None:
            parts.append(emb_landmark_anchors)
        parts.append(emb_positions)
        emb_all = torch.cat(parts, dim=1)
        emb_all = self.enc_layernorm(emb_all)
        emb_all = self.enc_dropout(emb_all)

        mask_pad = torch.zeros(
            emb_all.shape[:2],
            device=emb_all.device,
            dtype=torch.bool,
        )
        if length_anchors and landmark_anchor_mask is not None:
            anchor_start = (
                length_lang
                + emb_frames.shape[1]
                + emb_directions.shape[1]
                + emb_maps.shape[1]
            )
            mask_pad[
                :,
                anchor_start:anchor_start + length_anchors,
            ] = ~landmark_anchor_mask.bool()

        # Preserve the original HETT attention semantics: language queries see
        # only language tokens, while visual/map/anchor/grid tokens can attend
        # globally.  The only new restriction is key-padding on unused anchor
        # slots.
        total_length = emb_all.shape[1]
        mask_attn = torch.zeros(
            (total_length, total_length),
            device=emb_all.device,
            dtype=emb_all.dtype,
        )
        mask_attn[:length_lang, length_lang:] = float("-inf")

        output = self.enc_transformer(
            emb_all.transpose(0, 1),
            mask_attn,
            mask_pad,
        ).transpose(0, 1)
        return output, mask_pad

    def forward_with_centroids(
            self,
            emb_lang,
            emb_frames,
            emb_directions,
            emb_maps,
            emb_centroids,
            emb_positions,
            # lengths
    ):
        """
        pass embedded inputs through embeddings and encode them using a transformer
        """
        length_candidates = emb_positions.shape[1]
        # emb_lang is processed on each GPU separately so they size can vary
        length_lang = emb_lang.shape[1]
        length_frames = emb_frames.shape[1]
        length_centroids = emb_centroids.shape[1]


        # create a mask for padded elements
        length_mask_pad = length_lang + length_frames + 3 + length_candidates
        mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        # for i, l in enumerate(lengths):
        #     # mask padded frames
        #     mask_pad[i, (length_lang + l):(length_lang + length_max)] = True
        #     # mask padded directions
        #     mask_pad[i, (length_lang + length_max + l):] = True

        # encode the inputs
        emb_all = self.encode_inputs_centroids(emb_lang, emb_frames, emb_directions, emb_maps, emb_centroids, emb_positions, length_lang)


        # create a mask for attention (prediction at t should not see frames at >= t+1)
        mask_attn = model_util.generate_attention_mask(
            length_lang,
            length_frames,
            length_candidates,
            # length_centroids,
            emb_all.device,
        )

        # print(emb_all.shape, mask_attn.shape, mask_pad.shape)

        # encode the inputs
        output = self.enc_transformer(emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
        return output, mask_pad

    def encode_inputs(self, emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, lengths_lang):
        """
        add encodings (positional, token and so on)
        """

        emb_lang, emb_frames, emb_directions, emb_maps, emb_positions = self.enc_pos_map(
            emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, lengths_lang
        )

        emb_cat = torch.cat((emb_lang, emb_frames, emb_directions, emb_maps, emb_positions), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        emb_cat = self.enc_dropout(emb_cat)
        return emb_cat

    def encode_inputs_centroids(self, emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, emb_centroids, lengths_lang):
        """
        add encodings (positional, token and so on)
        """

        emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, emb_centroids = self.enc_pos_can(
            emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, emb_centroids, lengths_lang
        )

        emb_cat = torch.cat((emb_lang, emb_frames, emb_directions, emb_maps, emb_positions, emb_centroids), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        emb_cat = self.enc_dropout(emb_cat)
        return emb_cat


# class EncoderOBJ(nn.Module):
#     def __init__(self, args):
#         """
#         transformer encoder for language, frames and action inputs
#         """
#         super(EncoderOBJ, self).__init__()
#
#         # transofmer layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             args.demb,
#             args.encoder_heads,
#             args.demb,
#             args.dropout_transformer_encoder,
#         )
#         self.enc_transformer = nn.TransformerEncoder(encoder_layer, args.encoder_layers)
#
#         # how many last actions to attend to
#         self.num_input_actions = args.num_input_actions
#
#         # encodings
#         self.enc_pos = ObjPosEncoding(args.demb)
#         self.enc_pos_learn = None
#         self.enc_token = None
#         self.enc_layernorm = nn.LayerNorm(args.demb)
#         self.enc_dropout = nn.Dropout(args.dropout_emb, inplace=True)
#
#     def forward(
#             self,
#             emb_lang,
#             emb_frames,
#             emb_directions,
#             emb_objects,
#             obj_num,
#             lengths
#     ):
#         """
#         pass embedded inputs through embeddings and encode them using a transformer
#         """
#         length_max = np.max(lengths)
#         object_max = np.max(obj_num)
#         # emb_lang is processed on each GPU separately so they size can vary
#         length_lang = emb_lang.shape[1]
#         # create a mask for padded elements
#         length_mask_pad = length_lang + length_max * 2 + object_max
#         mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
#         for i, l in enumerate(lengths):
#             # mask padded frames
#             mask_pad[i, (length_lang + l):(length_lang + length_max)] = True
#             # mask padded directions
#             mask_pad[i, (length_lang + length_max + l):(length_lang + length_max * 2)] = True
#
#             mask_pad[i, (length_lang + length_max * 2 + obj_num[i]):] = True
#
#         # encode the inputs
#         emb_all = self.encode_inputs(emb_lang, emb_frames, emb_directions, emb_objects, length_lang, mask_pad)
#
#         # create a mask for attention (prediction at t should not see frames at >= t+1)
#         mask_attn = model_util.generate_attention_mask(
#             length_lang,
#             length_max,
#             object_max,
#             emb_all.device,
#         )
#
#         # encode the inputs
#         # print(emb_all.shape, mask_attn.shape, mask_pad.shape)
#         output = self.enc_transformer(emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
#         return output, mask_pad
#
#
#     def encode_inputs(self, emb_lang, emb_frames, emb_directions, emb_objects, lengths_lang, mask_pad):
#         """
#         add encodings (positional, token and so on)
#         """
#
#         emb_lang, emb_frames, emb_directions, emb_objects = self.enc_pos(
#             emb_lang, emb_frames, emb_directions, emb_objects, lengths_lang
#         )
#
#         emb_cat = torch.cat((emb_lang, emb_frames, emb_directions, emb_objects), dim=1)
#         emb_cat = self.enc_layernorm(emb_cat)
#         emb_cat = self.enc_dropout(emb_cat)
#         return emb_cat
