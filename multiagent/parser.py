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
    parser.add_argument('--heatmap_sigma', type=float, default=0.8,
                        help='Gaussian sigma (in grid cells) for Stage-1 heatmap supervision')
    parser.add_argument('--heatmap_loss_weight', type=float, default=0.1,
                        help='Weight of the Stage-1 heatmap loss')
    parser.add_argument('--trajectory_top_k', type=int, default=8)
    parser.add_argument('--trajectory_nms_kernel', type=int, default=3)
    parser.add_argument('--trajectory_horizons_m', type=float, nargs='+',
                        default=[10.0, 25.0, 50.0, 100.0],
                        help='Fixed physical horizons; final endpoint is appended automatically')
    parser.add_argument('--trajectory_loss_weight', type=float, default=1.0)
    parser.add_argument('--semantic_anchor_loss_weight', type=float, default=0.5,
                        help='Weight for next human core-anchor region classification')
    parser.add_argument('--semantic_anchor_position_loss_weight', type=float, default=1.0,
                        help='Weight for continuous core-anchor refinement')
    parser.add_argument('--semantic_anchor_goal_prior_scale', type=float, default=0.25,
                        help='Detached final-goal logit prior used by the next-core-anchor classifier')
    parser.add_argument('--human_anchor_min_step_m', type=float, default=1.0)
    parser.add_argument('--human_anchor_rdp_tolerance_m', type=float, default=2.5)
    parser.add_argument('--human_anchor_yaw_keyframe_deg', type=float, default=30.0)
    parser.add_argument('--human_anchor_landmark_radius_m', type=float, default=30.0)
    parser.add_argument('--human_anchor_min_lookahead_m', type=float, default=8.0)
    parser.add_argument('--student_anchor_max_projection_error_m', type=float, default=30.0,
                        help='Mask ambiguous semantic-anchor labels when a student state is far from the human path')
    parser.add_argument('--trajectory_residual_scale', type=float, default=0.05)
    parser.add_argument('--trajectory_execution_waypoint_index', type=int, default=1,
                        help='Waypoint executed from the predicted trajectory; 1 corresponds to the 25m horizon')
    parser.add_argument('--trajectory_move_iteration', type=int, default=10,
                        help='Low-level controller budget for reaching the selected trajectory waypoint')
    parser.add_argument('--stop_progress_threshold', type=float, default=0.95,
                        help='Student stop threshold; kept at the prior HETT value for baseline-preserving recovery')
    parser.add_argument('--semantic_policy_warmup_epochs', type=int, default=2,
                        help='Epochs that execute the stable final-goal trajectory prior before semantic-anchor control')
    parser.add_argument('--semantic_policy_ramp_epochs', type=int, default=3,
                        help='Epochs used to ramp semantic-anchor execution from 0 to 1')
    parser.add_argument('--student_rollout_warmup_epochs', type=int, default=2,
                        help='Teacher-only epochs before student rollout is introduced')
    parser.add_argument('--student_rollout_ramp_epochs', type=int, default=3,
                        help='Epochs used to ramp the student rollout weight')
    parser.add_argument('--student_rollout_max_weight', type=float, default=None,
                        help='Maximum student rollout weight; defaults to ml_weight')
    parser.add_argument('--min_initial_val_seen_sr', type=float, default=20.0,
                        help='Regression guard: abort after first val_seen evaluation if SR is below this value')
    parser.add_argument('--disable_initial_sr_guard', action='store_true', default=False,
                        help='Disable the first-epoch val_seen SR regression guard')
    parser.add_argument('--language_lr_scale', type=float, default=0.2,
                        help='BERT learning-rate multiplier relative to learning_rate')
    parser.add_argument('--vision_lr_scale', type=float, default=0.5,
                        help='DarkNet learning-rate multiplier relative to learning_rate')
    parser.add_argument('--pretrained_grad_clip', type=float, default=10.0,
                        help='Gradient-norm clip for trainable BERT and DarkNet encoders')
    parser.add_argument('--teacher_step_m', type=float, default=10.0,
                        help='Metric arc-length step for human teacher rollout')
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
    parser.add_argument('--disable_global_landmark_prior', action='store_true', default=False)
    parser.add_argument('--disable_referenced_landmark_mask', action='store_true', default=False)
    parser.add_argument('--enable_referenced_landmark_centroids', action='store_true', default=False,
                        help='Opt in to referenced-landmark centroid tokens; off by default')
    parser.add_argument('--disable_referenced_landmark_centroids', action='store_true', default=False)
    parser.add_argument('--max_referenced_landmarks', type=int, default=8)
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
    parser.add_argument('--learning_rate', type=float, default=1.0e-04)
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
    args.trajectory_steps = len(args.trajectory_horizons_m) + 1
    if args.teacher_step_m <= 0:
        raise ValueError('teacher_step_m must be positive')
    if args.human_anchor_min_lookahead_m <= 0:
        raise ValueError('human_anchor_min_lookahead_m must be positive')
    if args.human_anchor_rdp_tolerance_m < 0:
        raise ValueError('human_anchor_rdp_tolerance_m must be non-negative')
    if args.human_anchor_landmark_radius_m < 0:
        raise ValueError('human_anchor_landmark_radius_m must be non-negative')
    if args.student_anchor_max_projection_error_m <= 0:
        raise ValueError('student_anchor_max_projection_error_m must be positive')
    if args.semantic_anchor_goal_prior_scale < 0:
        raise ValueError('semantic_anchor_goal_prior_scale must be non-negative')
    if not 0.0 <= args.stop_progress_threshold <= 1.0:
        raise ValueError('stop_progress_threshold must be in [0, 1]')
    if not 0 <= args.trajectory_execution_waypoint_index < args.trajectory_steps:
        raise ValueError('trajectory_execution_waypoint_index is out of range')
    if args.semantic_policy_warmup_epochs < 0 or args.semantic_policy_ramp_epochs < 0:
        raise ValueError('semantic policy curriculum epochs must be non-negative')
    if args.student_rollout_warmup_epochs < 0 or args.student_rollout_ramp_epochs < 0:
        raise ValueError('student rollout curriculum epochs must be non-negative')
    if args.student_rollout_max_weight is None:
        args.student_rollout_max_weight = args.ml_weight
    if args.student_rollout_max_weight < 0:
        raise ValueError('student_rollout_max_weight must be non-negative')
    if args.min_initial_val_seen_sr < 0:
        raise ValueError('min_initial_val_seen_sr must be non-negative')
    if args.language_lr_scale <= 0 or args.vision_lr_scale <= 0:
        raise ValueError('language/vision lr scales must be positive')
    if args.pretrained_grad_clip <= 0:
        raise ValueError('pretrained_grad_clip must be positive')
    if args.trajectory_nms_kernel % 2 != 1:
        raise ValueError('trajectory_nms_kernel must be odd')
    if args.grid_size != 7:
        print('WARNING: HETT heatmap was designed for grid_size=7; got', args.grid_size)

    return args