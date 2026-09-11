import argparse
from pathlib import Path
from multiagent.defaultpaths import PROJECT_ROOT, WEIGHTS_DIR, GOAL_PREDICTOR_CHECKPOINT_DIR
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
    parser.add_argument('--grid_size', type=int, default=5)
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

    parser.add_argument('--darknet_model_file', type=str, default=str(WEIGHTS_DIR / 'yolo_v3.cfg'))
    parser.add_argument('--darknet_weight_file', type=str, default=str(WEIGHTS_DIR / 'best.pt'))
    parser.add_argument('--output_dir', type=str, default=str(GOAL_PREDICTOR_CHECKPOINT_DIR))
    parser.add_argument('--direction_loss_weight', type=float, default=1.5)
    parser.add_argument('--goal_loss_weight', type=float, default=2.0)
    parser.add_argument('--progress_loss_weight', type=float, default=0.1)
    parser.add_argument('--target_loss_weight', type=float, default=0.1,
                        help='auxiliary grid loss from released code (not specified in paper)')
    parser.add_argument('--disable_task_interaction', action='store_true')
    parser.add_argument('--enable_stage_recovery', action='store_true',
                        help='allow cautious fine-to-coarse recovery using model predictions')
    parser.add_argument('--stage_enter_distance', type=float, default=5.0)
    parser.add_argument('--stage_recovery_distance', type=float, default=20.0)
    parser.add_argument('--stage_recovery_patience', type=int, default=2)
    parser.add_argument('--progress_stop_threshold', type=float, default=0.95)
    parser.add_argument('--progress_stop_patience', type=int, default=2)
    parser.add_argument('--enable_region_grounding', action='store_true',
                        help='retain 7x7 visual regions and supervise visible/outside grounding')
    parser.add_argument('--region_loss_weight', type=float, default=0.1)
    parser.add_argument('--grounding_ablation', choices=['none', 'shuffle_language', 'shuffle_visual'],
                        default='none', help='evaluation-only grounding sensitivity check')

    # logger
    parser.add_argument('--log_every', type=int, default=1)
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
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.01,
                        help='released AdamW code used its implicit default 0.01')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_episodes', type=int, default=0, help='cap episodes per split for smoke tests (0 = use all)')
    parser.add_argument(
        '--feedback', type=str, default='student',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument("--nss_w", type=float, default=1)
    parser.add_argument("--nss_r", type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1, help='')
    parser.add_argument('--learning_rate', type=float, default=1.0e-04)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--resume_optimizer", action="store_true", default=False)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--train_trajectory_type', type=str, choices=['sp', 'mturk', 'both'], default='mturk')
    parser.add_argument('--train_episode_sample_size', type=int, default=-1)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')

    # eval params
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_first', action='store_true', default=False)
    parser.add_argument('--max_action_len', type=int, default=20)
    parser.add_argument('--eval_client', type=str, choices=['crop', 'airsim'], default='crop')
    parser.add_argument('--success_dist', type=float, default=20.)
    # parser.add_argument('--success_iou', type=float, default=0.4)
    parser.add_argument('--move_iteration', type=int, default=10)
    # parser.add_argument('--progress_stop_val', type=float, default=0.75)
    # parser.add_argument('--eval_goal_selector', type=str, choices=['gdino', 'llava'], default='gdino')
    parser.add_argument('--gps_noise_scale', type=float, default=0.)


    args = parser.parse_args()
    if args.grad_accum < 1 or args.batch_size < 1 or args.max_episodes < 0:
        parser.error('grad_accum/batch_size must be positive; max_episodes must be nonnegative')
    if args.stage_enter_distance < 0 or args.stage_recovery_distance <= args.stage_enter_distance:
        parser.error('stage_recovery_distance must be greater than stage_enter_distance')
    if args.stage_recovery_patience < 1 or args.progress_stop_patience < 1:
        parser.error('stage recovery/stop patience must be positive')
    if args.region_loss_weight < 0:
        parser.error('region_loss_weight must be nonnegative')
    if args.mode == 'train' and args.grounding_ablation != 'none':
        parser.error('grounding_ablation is evaluation-only')
    if args.save_every < 1:
        parser.error('save_every must be positive')
    if args.checkpoint and not Path(args.checkpoint).is_absolute():
        args.checkpoint = str(PROJECT_ROOT / args.checkpoint)
    output_dir = Path(args.output_dir)
    args.output_dir = str((output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir).resolve())
    args = postprocess_args(args)

    return args

def postprocess_args(args):
    # ROOTDIR = args.root_dir

    args.map_shape = (args.map_size, args.map_size)
    args.map_pixels_per_meter = args.map_size / args.map_meters


    return args
