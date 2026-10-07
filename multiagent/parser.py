import argparse
from typing import Literal, Optional
from dataclasses import dataclass, asdict




# @dataclass
# class ExperimentArgs:
#
#     seed: int
#     mode: Literal['train', 'eval']
#     local_rank: int
#     world_size: int
#
#     # model: Literal['mgp', 'seq2seq_with_map', 'cma_with_map']
#
#     # logger
#     log_dir: str
#
#     # model
#     demb: int
#     encoder_heads: int
#     encoder_layers: int
#     dropout_transformer_encoder: int
#     num_input_actions: int
#     dropout_emb: int
#
#     # observation
#     map_size: int
#     map_meters: float
#     map_update_interval: int
#     max_depth: float
#     altitude: float
#     ablate: Literal['rgb', 'depth', 'tracking', 'landmark', 'gsam', '']
#     alt_env: Literal['flood', 'ground_fissure', '']
#
#
#     # training params
#     optim: str
#     weight_decay: float
#     feedback: str
#     epsilon: float
#     learning_rate: float
#     train_batch_size: int
#     epochs: int
#     checkpoint: Optional[str]
#     save_every: int
#     train_trajectory_type: Literal['sp', 'mturk', 'both']
#     train_episode_sample_size: int
#
#     # eval params
#     eval_every: int
#     log_every: int
#     eval_batch_size: int
#     eval_at_start: bool
#     eval_max_timestep: int
#     eval_client: Literal['crop', 'airsim']
#     success_dist: float
#     success_iou: float
#     move_iteration: int
#     progress_stop_val: float
#     # eval_goal_selector: Literal['gdino', 'llava']
#     gps_noise_scale: float
#
#     ignore_id: int
#
#     def to_dict(self):
#         return asdict(self)
#
#     @property
#     def map_shape(self):
#         return self.map_size, self.map_size
#
#     @property
#     def map_pixels_per_meter(self):
#         return self.map_size / self.map_meters



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'visualize'], default='train')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')

    # parser.add_argument('--model', type=str, choices=['mgp', 'seq2seq_with_map', 'cma_with_map'], default='mgp')
    parser.add_argument('--ignore_id', type=int, default=-100, help='ignoreid for action')

    # model
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--heatmap_grid_size', type=int, default=28,
                        help='Dense spatial-belief field size')
    parser.add_argument('--heatmap_sigma_m', type=float, default=20.0,
                        help='Metric Gaussian sigma in meters for belief supervision')
    parser.add_argument('--heatmap_top_k', type=int, default=16,
                        help='Number of NMS belief hypotheses retained per step')
    parser.add_argument('--heatmap_nms_kernel', type=int, default=3,
                        help='Odd NMS suppression kernel on the dense belief field')
    parser.add_argument('--heatmap_local_window', type=int, default=3,
                        help='Odd local window for soft-argmax refinement around Top-1')
    parser.add_argument('--heatmap_loss_weight', type=float, default=0.1,
                        help='Weight of the Stage-1 heatmap loss')
    parser.add_argument('--disable_reference_rerank', action='store_true', default=False,
                        help='Disable evidence-backed referenced-landmark Top-K reranking')
    parser.add_argument('--reference_rerank_top_k', type=int, default=4,
                        help='Only the first K NMS hypotheses may be reranked by the reference prior')
    parser.add_argument('--reference_rerank_weight', type=float, default=0.50,
                        help='Weight of the referenced-landmark proximity prior during conservative reranking')
    parser.add_argument('--reference_rerank_max_log_margin', type=float, default=0.20,
                        help='Maximum raw Top1-vs-runner-up log-probability margin that permits reranking')
    parser.add_argument('--reference_rerank_min_prior_gain', type=float, default=0.30,
                        help='Minimum reference-prior gain required to replace raw Top-1')
    parser.add_argument('--reference_prior_dilation_steps', type=int, default=1,
                        help='Number of dense-grid cells over which referenced-landmark support is expanded')
    parser.add_argument('--reference_prior_decay', type=float, default=0.70,
                        help='Per-cell decay for referenced-landmark proximity support')
    parser.add_argument('--stage1_switch_dist', type=float, default=25.0,
                        help='Distance in meters for switching from coarse Stage 1 to fine Stage 2')
    parser.add_argument('--stage2_recover_dist', type=float, default=40.0,
                        help='Distance in meters for returning from Stage 2 to coarse Stage 1')
    parser.add_argument('--stage2_recover_patience', type=int, default=2,
                        help='Consecutive far-away Stage-2 steps required before coarse recovery')
    parser.add_argument('--demb', type=int, default=768)
    parser.add_argument('--encoder_heads', type=int, default=12)
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument("--dropout_transformer_encoder",type=float, default=0.1)
    parser.add_argument('--num_input_actions', type=int, default=1)
    # dropout rate for processed lang and visual embeddings
    parser.add_argument("--dropout_emb",type=float, default=0)
    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_h_layers', type=int, default=0)
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--num_replacement', '-num_replacement', action='store_true', default=False)
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument("--teacher_weight", type=float, default=1.)

    parser.add_argument('--darknet_model_file', type=str, default='../weights/yolo_v3.cfg')
    parser.add_argument('--darknet_weight_file', type=str, default='../weights/best.pt')

    # logger
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='log')

    # observation
    parser.add_argument('--map_size', type=int, default=240)
    parser.add_argument('--map_meters', type=float, default=410.)
    parser.add_argument('--map_update_interval', type=int, default=5)
    parser.add_argument('--max_depth', type=float, default=200.)
    parser.add_argument('--altitude', type=float, default=50)
    parser.add_argument('--ablate', type=str, choices=['rgb', 'depth', 'tracking', 'landmark', 'gsam', ''], default='')
    parser.add_argument('--alt_env', type=str, choices=['', 'flood', 'ground_fissure'], default='')

    
    # training params
    parser.add_argument(
        '--optim', type=str, default='adam',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='student',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument("--nss_w", type=float, default=1)
    parser.add_argument("--nss_r", type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1, help='')
    parser.add_argument('--learning_rate', type=float, default=1.0e-03)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--benchmark_batches', type=int, default=0,
                        help='Stop after this many training batches and skip validation; 0 disables')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--resume_optimizer", action="store_true", default=False)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--train_trajectory_type', type=str, choices=['sp', 'mturk', 'both'], default='mturk')
    parser.add_argument('--train_episode_sample_size', type=int, default=-1)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')

    # eval params
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--eval_first', action='store_true', default=False)
    parser.add_argument('--max_action_len', type=int, default=20)
    parser.add_argument('--eval_client', type=str, choices=['crop', 'airsim'], default='crop')
    parser.add_argument('--success_dist', type=float, default=20.)
    # parser.add_argument('--success_iou', type=float, default=0.4)
    parser.add_argument('--move_iteration', type=int, default=5)
    # parser.add_argument('--progress_stop_val', type=float, default=0.75)
    # parser.add_argument('--eval_goal_selector', type=str, choices=['gdino', 'llava'], default='gdino')
    parser.add_argument('--gps_noise_scale', type=float, default=0.)


    args = parser.parse_args()
    args = postprocess_args(args)

    return args

def postprocess_args(args):
    # ROOTDIR = args.root_dir

    args.map_shape = (args.map_size, args.map_size)
    args.map_pixels_per_meter = args.map_size / args.map_meters


    return args
