import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import sys, os

from defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 先加入绝对路径，否则会报错，注意__file__表示的是当前执行文件的路径
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from agent import NavCMTAgent
from env import CityNavBatch
from parser import parse_args

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def _mean_log(logs, key):
    values = logs.get(key, [])
    return float(np.mean(values)) if values else None


DIAGNOSTIC_NAMES = (
    'heatmap_top1_acc',
    'heatmap_top3_acc',
    'heatmap_gt_prob',
    'heatmap_top1_conf',
    'heatmap_entropy',
    'heatmap_gt_rank',
    'heatmap_cell_error',
    'heatmap_coarse_goal_error_m',
    'semantic_anchor_top1_acc',
    'semantic_anchor_top3_acc',
    'semantic_anchor_gt_prob',
    'semantic_anchor_top1_conf',
    'semantic_anchor_entropy',
    'semantic_anchor_gt_rank',
    'semantic_anchor_cell_error',
    'semantic_anchor_coarse_goal_error_m',
    'human_anchor_distance_m',
    'human_anchor_projection_error_m',
    'human_anchor_turn_rate',
    'human_anchor_yaw_rate',
    'human_anchor_landmark_rate',
    'human_anchor_goal_rate',
    'trajectory_top1_endpoint_error_m',
    'trajectory_oracle_topk_endpoint_error_m',
    'trajectory_topk_gt_recall',
    'trajectory_first_wp_error_m',
    'progress_mae',
    'stop_trigger_rate',
    'premature_stop_rate',
    'near_goal_continue_rate',
)


def collect_policy_diagnostics(logs, policy):
    diagnostics = {}
    for name in DIAGNOSTIC_NAMES:
        value = _mean_log(logs, policy + '_' + name)
        if value is not None:
            diagnostics[name] = value

    if policy == 'teacher':
        for key in ('teacher_coverage_ratio', 'teacher_truncated_rate'):
            value = _mean_log(logs, key)
            if value is not None:
                diagnostics[key] = value
    return diagnostics


def collect_heatmap_distribution(logs, policy, grid_size):
    prefix = policy + '_'
    count_values = logs.get(prefix + 'heatmap_sample_count', [])
    if not count_values:
        return None

    sample_count = float(np.sum(count_values))
    if sample_count <= 0:
        return None

    gt_hist = np.sum(
        np.asarray(logs[prefix + 'heatmap_gt_hist']),
        axis=0,
    )
    pred_hist = np.sum(
        np.asarray(logs[prefix + 'heatmap_pred_hist']),
        axis=0,
    )
    prob_sum = np.sum(
        np.asarray(logs[prefix + 'heatmap_prob_sum']),
        axis=0,
    )
    confusion = np.sum(
        np.asarray(logs[prefix + 'heatmap_confusion']),
        axis=0,
    )

    cell_count = grid_size ** 2
    return {
        'grid_size': int(grid_size),
        'sample_count': sample_count,
        'gt_frequency': (gt_hist / sample_count).reshape(
            grid_size, grid_size
        ).tolist(),
        'pred_frequency': (pred_hist / sample_count).reshape(
            grid_size, grid_size
        ).tolist(),
        'mean_probability': (prob_sum / sample_count).reshape(
            grid_size, grid_size
        ).tolist(),
        'confusion_counts': confusion.reshape(
            cell_count, cell_count
        ).astype(np.int64).tolist(),
    }


def write_diagnostics_jsonl(
    path,
    epoch,
    phase,
    split,
    policy,
    diagnostics,
    metrics=None,
    heatmap_distribution=None,
):
    payload = {
        'epoch': int(epoch),
        'phase': phase,
        'split': split,
        'policy': policy,
        'diagnostics': diagnostics,
    }
    if metrics is not None:
        payload['navigation_metrics'] = {
            key: float(value)
            for key, value in metrics.items()
            if np.isscalar(value) and np.isfinite(value)
        }
    if heatmap_distribution is not None:
        payload['heatmap_distribution'] = heatmap_distribution
    with open(path, 'a') as outf:
        outf.write(json.dumps(payload, sort_keys=True) + '\n')


def format_diagnostics_line(policy, diagnostics):
    preferred = [
        'heatmap_top1_acc',
        'heatmap_top3_acc',
        'heatmap_gt_prob',
        'heatmap_top1_conf',
        'heatmap_entropy',
        'heatmap_gt_rank',
        'heatmap_cell_error',
        'heatmap_coarse_goal_error_m',
        'semantic_anchor_top1_acc',
        'semantic_anchor_top3_acc',
        'semantic_anchor_gt_rank',
        'semantic_anchor_coarse_goal_error_m',
        'human_anchor_distance_m',
        'human_anchor_turn_rate',
        'human_anchor_yaw_rate',
        'human_anchor_landmark_rate',
        'human_anchor_goal_rate',
        'trajectory_top1_endpoint_error_m',
        'trajectory_oracle_topk_endpoint_error_m',
        'trajectory_topk_gt_recall',
        'trajectory_first_wp_error_m',
        'progress_mae',
        'stop_trigger_rate',
        'premature_stop_rate',
        'near_goal_continue_rate',
        'teacher_coverage_ratio',
        'teacher_truncated_rate',
    ]
    parts = [
        '%s=%.4f' % (key, diagnostics[key])
        for key in preferred
        if key in diagnostics
    ]
    return 'DIAG %s %s' % (policy, ' '.join(parts))


def get_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/cver/xcding/code/tokenizer_files/bert-base-uncase')
    return tokenizer


def build_train_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch

    train_env = dataset_class(
        'train_seen',
        args,
        batch_size=args.batch_size,
        seed=args.seed + rank,
        rank=rank,
        world_size=args.world_size
    )

    val_env_names = ['val_seen', 'val_unseen', ]  # 'test_unseen'
    # val_env_names = ['val_seen',]  # 'test_unseen'

    val_envs = {}
    for split in val_env_names:
        val_env = dataset_class(
            split, args,
            batch_size=args.batch_size,
            seed=args.seed + rank,
            rank=rank,
            world_size=1
        )

        val_envs[split] = val_env

    return train_env, val_envs


def build_val_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch

    val_env_names = ['val_seen', 'val_unseen', 'test_unseen', ]  # 'test_unseen'
    # val_env_names = ['visualization' ]  # 'test_unseen'

    val_envs = {}
    for split in val_env_names:
        val_env = dataset_class(
            split, args,
            batch_size=args.batch_size,
            seed=args.seed + rank,
            rank=rank,
            world_size=1
        )

        val_envs[split] = val_env

    return val_envs

def build_vis_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch
    #
    # val_env_names = ['val_seen', 'val_unseen', 'test_unseen', ]  # 'test_unseen'
    val_env_names = ['visualization' ]  # 'test_unseen'

    val_envs = {}
    for split in val_env_names:
        val_env = dataset_class(
            split, args,
            batch_size=args.batch_size,
            seed=args.seed + rank,
            rank=rank,
            world_size=1
        )

        val_envs[split] = val_env

    return val_envs


def train(args, train_env, val_envs, rank=-1):
    # print('?')
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        # writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'train.txt')
        diagnostics_file = os.path.join(
            GOAL_PREDICTOR_CHECKPOINT_DIR,
            'navigation_diagnostics.jsonl',
        )
        write_to_record_file(str(args) + '\n\n', record_file)

    best_val = {'val_unseen': {"sr": 0., "state": ""}, 'val_unseen_full_traj': {"sr": 0., "state": ""}}

    # first evaluation
    if args.eval_first:
        loss_str = ""
        start_epoch = -1
        if default_gpu:

            for env_name, env in val_envs.items():
                agent_class_eval = NavCMTAgent
                agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)

                if args.checkpoint is not None:
                    start_epoch = agent_eval.load(os.path.join(args.checkpoint))
                    if default_gpu:
                        write_to_record_file(
                            "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
                            record_file
                        )

                agent_eval.env = env
                # sampler = DistributedSampler(env, num_replicas=args.world_size, rank=rank)
                loader = DataLoader(env, batch_size=1)
                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, env_name=env_name, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s \n" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                if env_name in best_val:
                    if score_summary['sr'] >= best_val[env_name]['sr']:
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Epoch %d %s' % (start_epoch, loss_str)
            write_to_record_file(loss_str, record_file)

    torch.cuda.empty_cache()
    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)

    # resume file
    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch = agent.load(os.path.join(args.checkpoint))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
                record_file
            )

    # Start Training
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start epoch: %s' % str(start_epoch), record_file
        )



    torch.cuda.empty_cache()
    # interval = int(train_env.size() / args.batch_size) * args.log_every

    # zero_start_iter = 0
    for idx in range(start_epoch, args.epochs):
        agent.logs = defaultdict(list)

        # iter = idx + interval
        # if args.train_val_on_full:
        #     agent.env = train_full_traj_env
        # else:
        agent.env = train_env
        # print(agent.env.size())
        loader = DataLoader(agent.env, batch_size=1)
        # print(loader.dataset.size())

        # Train for 2 epochs before evaluate again
        agent.train(loader, args.log_every, feedback=args.feedback,
                    nss_w_weighting=1)  # nss_w_weighting = max(0, (args.iters/2 - idx)/ (args.iters/2)))

        if args.benchmark_batches:
            if default_gpu:
                agent.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest"))
                ml_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)
                direction_loss = sum(agent.logs['direction_loss']) / max(len(agent.logs['direction_loss']), 1)
                progress_loss = sum(agent.logs['progress_loss']) / max(len(agent.logs['progress_loss']), 1)
                goal_predict_loss = sum(agent.logs['goal_predict_loss']) / max(
                    len(agent.logs['goal_predict_loss']), 1
                )
                heatmap_loss = sum(agent.logs['heatmap_loss']) / max(
                    len(agent.logs['heatmap_loss']), 1
                )
                semantic_anchor_loss = sum(
                    agent.logs['semantic_anchor_loss']
                ) / max(len(agent.logs['semantic_anchor_loss']), 1)
                semantic_anchor_position_loss = sum(
                    agent.logs['semantic_anchor_position_loss']
                ) / max(len(agent.logs['semantic_anchor_position_loss']), 1)
                trajectory_loss = sum(agent.logs['trajectory_loss']) / max(
                    len(agent.logs['trajectory_loss']), 1
                )
                print(
                    "BENCHMARK_EPOCH epoch=%d IL_loss=%.6f direction_loss=%.6f "
                    "progress_loss=%.6f goal_predict_loss=%.6f heatmap_loss=%.6f "
                    "semantic_anchor_loss=%.6f semantic_anchor_position_loss=%.6f "
                    "trajectory_loss=%.6f" % (
                        idx,
                        ml_loss,
                        direction_loss,
                        progress_loss,
                        goal_predict_loss,
                        heatmap_loss,
                        semantic_anchor_loss,
                        semantic_anchor_position_loss,
                        trajectory_loss,
                    ),
                    flush=True,
                )
                for policy in ('teacher', 'student'):
                    diagnostics = collect_policy_diagnostics(agent.logs, policy)
                    if diagnostics:
                        print(
                            format_diagnostics_line(policy, diagnostics),
                            flush=True,
                        )
            torch.cuda.empty_cache()
            continue

        if default_gpu:
            ml_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)

            direction_loss = sum(agent.logs['direction_loss']) / max(len(agent.logs['direction_loss']), 1)

            progress_loss = sum(agent.logs['progress_loss']) / max(len(agent.logs['progress_loss']), 1)
            goal_predict_loss = sum(agent.logs['goal_predict_loss']) / max(len(agent.logs['goal_predict_loss']), 1)
            semantic_anchor_loss = (
                sum(agent.logs['semantic_anchor_loss'])
                / max(len(agent.logs['semantic_anchor_loss']), 1)
            )
            semantic_anchor_position_loss = (
                sum(agent.logs['semantic_anchor_position_loss'])
                / max(len(agent.logs['semantic_anchor_position_loss']), 1)
            )
            trajectory_loss = (
                sum(agent.logs['trajectory_loss'])
                / max(len(agent.logs['trajectory_loss']), 1)
            )

            write_to_record_file(
                "\nIL_loss %.4f direction_loss %.4f progress_loss %.4f "
                "goal_predict_loss %.4f semantic_anchor_loss %.4f "
                "semantic_anchor_position_loss %.4f trajectory_loss %.4f" % (
                    ml_loss,
                    direction_loss,
                    progress_loss,
                    goal_predict_loss,
                    semantic_anchor_loss,
                    semantic_anchor_position_loss,
                    trajectory_loss,
                ),
                record_file
            )
            trajectory_step = (
                sum(agent.logs['trajectory_step'])
                / max(len(agent.logs['trajectory_step']), 1)
            )
            teacher_step = (
                sum(agent.logs['teacher_step'])
                / max(len(agent.logs['teacher_step']), 1)
            )
            teacher_distance_m = (
                sum(agent.logs['teacher_distance_m'])
                / max(len(agent.logs['teacher_distance_m']), 1)
            )
            global_landmark_gate = (
                sum(agent.logs['global_landmark_gate'])
                / max(len(agent.logs['global_landmark_gate']), 1)
            )
            semantic_anchor_landmark_gate = _mean_log(
                agent.logs, 'semantic_anchor_landmark_gate'
            ) or 0.0
            semantic_anchor_visual_gate = _mean_log(
                agent.logs, 'semantic_anchor_visual_gate'
            ) or 0.0
            semantic_anchor_language_gate = _mean_log(
                agent.logs, 'semantic_anchor_language_gate'
            ) or 0.0
            write_to_record_file(
                "\nrollout trajectory_step %.4f teacher_step %.4f "
                "teacher_distance_m %.4f global_landmark_gate %.4f "
                "anchor_gates landmark %.4f visual %.4f language %.4f" % (
                    trajectory_step,
                    teacher_step,
                    teacher_distance_m,
                    global_landmark_gate,
                    semantic_anchor_landmark_gate,
                    semantic_anchor_visual_gate,
                    semantic_anchor_language_gate,
                ),
                record_file
            )

            for policy in ('teacher', 'student'):
                diagnostics = collect_policy_diagnostics(agent.logs, policy)
                if diagnostics:
                    write_to_record_file(
                        "\n" + format_diagnostics_line(policy, diagnostics),
                        record_file,
                    )
                    heatmap_distribution = collect_heatmap_distribution(
                        agent.logs,
                        policy,
                        args.grid_size,
                    )
                    write_diagnostics_jsonl(
                        diagnostics_file,
                        idx,
                        'train',
                        'train_seen',
                        policy,
                        diagnostics,
                        heatmap_distribution=heatmap_distribution,
                    )

            # Run validation
            loss_str = "\nepoch {}".format(idx)

            agent.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest"))
            agent_class_eval = NavCMTAgent
            agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
            print("Loaded the listener model at epoch %d from %s" % \
                  (agent_eval.load(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest")),
                   os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest")))
            for env_name, env in val_envs.items():
                agent_eval.logs = defaultdict(list)
                agent_eval.env = env
                loader = DataLoader(env, batch_size=1)
                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, env_name=env_name, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                trajectory_step = (
                    sum(agent_eval.logs['trajectory_step'])
                    / max(len(agent_eval.logs['trajectory_step']), 1)
                )
                global_landmark_gate = (
                    sum(agent_eval.logs['global_landmark_gate'])
                    / max(len(agent_eval.logs['global_landmark_gate']), 1)
                )
                anchor_landmark_gate = _mean_log(
                    agent_eval.logs, 'semantic_anchor_landmark_gate'
                ) or 0.0
                anchor_visual_gate = _mean_log(
                    agent_eval.logs, 'semantic_anchor_visual_gate'
                ) or 0.0
                anchor_language_gate = _mean_log(
                    agent_eval.logs, 'semantic_anchor_language_gate'
                ) or 0.0
                write_to_record_file(
                    "\nrollout trajectory_step %.4f global_landmark_gate %.4f "
                    "anchor_gates landmark %.4f visual %.4f language %.4f" % (
                        trajectory_step,
                        global_landmark_gate,
                        anchor_landmark_gate,
                        anchor_visual_gate,
                        anchor_language_gate,
                    ),
                    record_file
                )
                diagnostics = collect_policy_diagnostics(
                    agent_eval.logs,
                    'student',
                )
                if diagnostics:
                    write_to_record_file(
                        "\n" + format_diagnostics_line('student', diagnostics),
                        record_file,
                    )
                    heatmap_distribution = collect_heatmap_distribution(
                        agent_eval.logs,
                        'student',
                        args.grid_size,
                    )
                    write_diagnostics_jsonl(
                        diagnostics_file,
                        idx,
                        'validation',
                        env_name,
                        'student',
                        diagnostics,
                        score_summary,
                        heatmap_distribution,
                    )
                loss_str += "\n%s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                if env_name in best_val:
                    if score_summary['sr'] >= best_val[env_name]['sr']:
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Epoch %d %s' % (idx, loss_str)
                        agent_eval.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "best_%s" % (env_name)))

            write_to_record_file(
                ('\n%s (%d %d%%) %s' % (
                    timeSince(start, float(idx + 1) / args.epochs), idx + 1, float(idx + 1) / args.epochs * 100,
                    loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)
        torch.cuda.empty_cache()


def valid(args, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)
    if default_gpu:

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = -1
        loss_str = "\nepoch {}".format(epoch)
        if args.checkpoint is not None:
            epoch = agent_eval.load(args.checkpoint)
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'valid.txt')
        diagnostics_file = os.path.join(
            GOAL_PREDICTOR_CHECKPOINT_DIR,
            'navigation_diagnostics.jsonl',
        )
        for env_name, env in val_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.test(loader, env_name=env_name, feedback='student')
            pred_results = agent_eval.get_results()

            score_summary, result = env.eval_metrics(pred_results)
            trajectory_step = (
                sum(agent_eval.logs['trajectory_step'])
                / max(len(agent_eval.logs['trajectory_step']), 1)
            )
            write_to_record_file(
                "\nrollout trajectory_step %.4f" % trajectory_step,
                record_file
            )
            diagnostics = collect_policy_diagnostics(
                agent_eval.logs,
                'student',
            )
            if diagnostics:
                write_to_record_file(
                    "\n" + format_diagnostics_line('student', diagnostics),
                    record_file,
                )
                heatmap_distribution = collect_heatmap_distribution(
                    agent_eval.logs,
                    'student',
                    args.grid_size,
                )
                write_diagnostics_jsonl(
                    diagnostics_file,
                    epoch,
                    'validation',
                    env_name,
                    'student',
                    diagnostics,
                    score_summary,
                    heatmap_distribution,
                )
            loss_str += "\n%s " % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
        write_to_record_file(
            ('\n%s' % loss_str),
            record_file
        )
        # json.dump(
        #     result,
        #     open(os.path.join(args.pred_dir, "eval_detail_%s.json" % env_name), 'w'),
        #     sort_keys=True, indent=4, separators=(',', ': ')
        # )

def visualize(args, vis_envs, rank=-1):
    default_gpu = is_default_gpu(args)
    if default_gpu:

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = agent_eval.load(args.checkpoint)
        if args.checkpoint is not None:
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'valid.txt')
        for env_name, env in vis_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.visualize(loader, feedback='student')
            pred_results = agent_eval.get_results()

        #     score_summary, result = env.eval_metrics(pred_results)
        #     stage1_step = sum(agent_eval.logs['stage1_step']) / max(len(agent_eval.logs['stage1_step']), 1)
        #     stage2_step = sum(agent_eval.logs['stage2_step']) / max(len(agent_eval.logs['stage2_step']), 1)
        #     stage2_rotate = sum(agent_eval.logs['stage2_rotate']) / max(len(agent_eval.logs['stage2_rotate']), 1)
        #
        #     write_to_record_file(
        #         "\nstage %.4f %.4f %.4f" % (
        #             stage1_step, stage2_step, stage2_rotate),
        #         record_file
        #     )
        #     loss_str += "\n%s " % env_name
        #     for metric, val in score_summary.items():
        #         loss_str += ', %s: %.2f' % (metric, val)
        #         # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
        # write_to_record_file(
        #     ('\n%s' % loss_str),
        #     record_file
        # )
        # json.dump(
        #     result,
        #     open(os.path.join(args.pred_dir, "eval_detail_%s.json" % env_name), 'w'),
        #     sort_keys=True, indent=4, separators=(',', ': ')
        # )


def main():
    args = parse_args()
    rank = 0
    # if args.train_val_on_full:
    #     args.max_action_len *= 4
    if args.world_size > 1:
        rank = init_distributed(args)
        # print('success')
        args.local_rank = rank
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0
    # if args.vision_only:
    #     print("!!! Vision only")
    # if args.language_only:
    #     print("!!! Language only")

    set_random_seed(args.seed + rank)

    if args.mode == 'train':
        train_env, val_envs = build_train_dataset(args, rank=rank)
        train(args, train_env, val_envs, rank=rank)
    elif args.mode == 'eval':
        val_envs = build_val_dataset(args, rank=rank)
        valid(args, val_envs, rank=rank)
    elif args.mode == 'visualize':
        vis_envs = build_vis_dataset(args, rank=rank)
        visualize(args, vis_envs, rank=rank)


if __name__ == '__main__':
    main()
