from tqdm import tqdm

from multiagent.cityreferobject import MultiMapObjects
from multiagent.dataset.episode import Episode
from multiagent.teacher.algorithm.lookahead import LookaheadTeacherParams
from multiagent.teacher.trajectory import TeacherParams, TeacherType, get_teacher_actions_and_trajectory
from multiagent.trajectory import TrajectoryType, trajectory_registry
from multiagent.dataset.mturk_trajectory import MTurkTrajectory
from typing import List, Dict, Callable
import Levenshtein

from multiagent.actions import DiscreteAction
from multiagent.teacher.preprocess import optimize_teacher_path


def _description_landmark_contours(objects, mturk_traj):
    target = objects[mturk_traj.map_name][mturk_traj.object_id]
    names = target.processed_descriptions[mturk_traj.desc_id].landmarks
    candidates = [obj for obj in objects[mturk_traj.map_name].values() if obj.name]
    contours = []
    for name in names:
        matches = [obj for obj in candidates if obj.name.lower().strip() == name.lower().strip()]
        landmark = max(matches, key=lambda obj: obj.area) if matches else min(
            candidates, key=lambda obj: Levenshtein.distance(obj.name, name))
        contours.append(landmark.contour)
    return contours

def generate_episodes_from_mturk_trajectories(
    objects: MultiMapObjects,
    mturk_trajectories: List[MTurkTrajectory],
    max_dist_marker_to_target=30,
    max_steps=500,
    teacher_type: TeacherType = 'lookahead',
    teacher_params: TeacherParams = LookaheadTeacherParams(lookahead=1),
    optimize_landmark_teacher=False,
    landmark_arrival_radius=20.0,
    teacher_coarse_moves=10,
    teacher_local_moves=10,
) -> List[Episode]:
    episodes = []
    for mturk_traj in tqdm(mturk_trajectories, desc='generating episodes'):

        if mturk_traj.dist_marker_to_target > max_dist_marker_to_target:
            continue

        optimized = None
        if optimize_landmark_teacher:
            optimized = optimize_teacher_path(
                mturk_traj.interpolated_pose_trajectory,
                _description_landmark_contours(objects, mturk_traj),
                arrival_radius=landmark_arrival_radius,
                coarse_moves=teacher_coarse_moves,
                local_moves=teacher_local_moves,
            )
        if optimized is None:
            teacher_actions, teacher_trajectory = get_teacher_actions_and_trajectory(
                teacher_type, teacher_params, mturk_traj.start_pose, mturk_traj.interpolated_trajectory
            )
            stage_boundary = None
        else:
            teacher_trajectory = optimized.poses
            teacher_actions = [DiscreteAction.MOVE_FORWARD.index] * (len(teacher_trajectory) - 1)
            teacher_actions.append(DiscreteAction.STOP.index)
            stage_boundary = optimized.stage_boundary

        if len(teacher_actions) <= max_steps:
            episodes.append(Episode(
                objects[mturk_traj.map_name][mturk_traj.object_id], mturk_traj.desc_id,
                teacher_trajectory, teacher_actions,
                teacher_stage_boundary=stage_boundary,
                teacher_optimized=optimized is not None,
            ))
    
    return episodes


def convert_trajectory_to_shortest_path(
    episode: Episode,
    trajectory_type: TrajectoryType = 'linear_xy',
    use_teacher_dst=False,
    teacher_type: TeacherType = 'lookahead',
    teacher_params: TeacherParams = LookaheadTeacherParams(),
) -> Episode:
    dst = episode.teacher_trajectory[-1].xyz if use_teacher_dst else episode.target_position
    sp_trajectory = trajectory_registry[trajectory_type](episode.start_pose.xyz, dst)
    teacher_actions, teacher_trajectory = get_teacher_actions_and_trajectory(
        teacher_type, teacher_params, episode.start_pose, sp_trajectory
    )

    return Episode(
        episode.target_object,
        episode.description_id,
        teacher_trajectory,
        teacher_actions,
    )
