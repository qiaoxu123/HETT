from dataclasses import dataclass
import math

from multiagent.cityreferobject import CityReferObject
from multiagent.space import Pose4D
import numpy as np

from typing import List, Dict, Tuple

MapName = str
ObjectID = int
DescriptionID = int
EpisodeID = Tuple[MapName, ObjectID, DescriptionID]


@dataclass
class Episode:
    target_object: CityReferObject
    description_id: int
    teacher_trajectory: List[Pose4D]
    teacher_actions: List[int]

    @property
    def description_landmarks(self):
        return self.target_object.processed_descriptions[self.description_id].landmarks
    
    @property
    def description_surroundings(self):
        return self.target_object.processed_descriptions[self.description_id].surroundings

    @property
    def description_target(self):
        return self.target_object.processed_descriptions[self.description_id].target

    @property
    def id(self) -> EpisodeID:
        return self.map_name, self.target_object.id, self.description_id

    @property
    def map_name(self):
        return self.target_object.map_name
    
    @property
    def start_pose(self):
        return self.teacher_trajectory[0]
    
    @property
    def target_description(self):
        return self.target_object.descriptions[self.description_id]

    @property
    def target_position(self):
        return self.target_object.position
    
    @property
    def target_processed_description(self):
        return self.target_object.processed_descriptions[self.description_id]

    @property
    def target_type(self):
        return self.target_object.object_type

    @property
    def time_step(self):
        return len(self.teacher_actions)
    
    @property
    def trajectory(self):
        return self.teacher_trajectory
    
    def sample_trajectory(self, interval: int, end_idx: int):
        return self.teacher_trajectory[:end_idx][::interval] + [self.teacher_trajectory[end_idx]]

    def sample_actions(self, interval: int, end_idx: int):

        return self.teacher_actions[:end_idx][::interval] + [self.teacher_actions[end_idx]]


_COMPASS_DIRECTIONS = (
    'east', 'northeast', 'north', 'northwest',
    'west', 'southwest', 'south', 'southeast',
)


def metric_direction_name(dx: float, dy: float) -> str:
    """Return an eight-way world-frame direction for a displacement."""
    angle = math.atan2(dy, dx)
    direction_index = int(round(angle / (math.pi / 4))) % 8
    return _COMPASS_DIRECTIONS[direction_index]


def reverse_flight_trajectory(trajectory: List[Pose4D]) -> List[Pose4D]:
    """Reverse positions and point every camera along the reverse motion.

    Merely reversing RGB/camera poses leaves the camera facing the old travel
    direction.  Recomputing yaw makes the rendered observation agree with the
    supervised reverse motion.  The final pose reuses the last valid reverse
    heading because it has no successor.
    """
    reversed_poses = list(reversed(trajectory))
    if len(reversed_poses) < 2:
        return reversed_poses

    headings = []
    last_heading = reversed_poses[0].yaw
    for current, following in zip(reversed_poses[:-1], reversed_poses[1:]):
        dx = following.x - current.x
        dy = following.y - current.y
        if math.hypot(dx, dy) > 1e-6:
            last_heading = math.atan2(dy, dx)
        headings.append(last_heading)
    headings.append(last_heading)
    return [
        Pose4D(pose.x, pose.y, pose.z, yaw)
        for pose, yaw in zip(reversed_poses, headings)
    ]


@dataclass(frozen=True)
class ReverseTeacherEpisode:
    """Training-only reverse view of a human demonstration.

    The reverse task starts at the final human pose, follows the recorded human
    poses backwards, and uses the original start as its metric destination.
    Target semantics and landmark instances still come from the original
    instruction, but no reverse state is ever reused by the forward student.
    """

    forward_episode: Episode

    @property
    def target_object(self):
        return self.forward_episode.target_object

    @property
    def description_id(self):
        return self.forward_episode.description_id

    @property
    def description_landmarks(self):
        return self.forward_episode.description_landmarks

    @property
    def description_surroundings(self):
        return self.forward_episode.description_surroundings

    @property
    def description_target(self):
        return self.forward_episode.description_target

    @property
    def id(self):
        return self.forward_episode.id

    @property
    def map_name(self):
        return self.forward_episode.map_name

    @property
    def start_pose(self):
        return self.forward_episode.teacher_trajectory[-1]

    @property
    def target_position(self):
        # The original route start is the metric destination of the reverse task.
        return self.forward_episode.start_pose.xyz

    @property
    def teacher_trajectory(self):
        return reverse_flight_trajectory(self.forward_episode.teacher_trajectory)

    @property
    def target_view_poses(self):
        """Original human views kept separate from reverse action frames."""
        return self.forward_episode.teacher_trajectory

    @property
    def visual_alignment_text(self):
        # The parsed target phrase deliberately excludes context landmarks.
        # Those remain separate inputs to the forward multi-landmark head.
        target = self.forward_episode.description_target.strip()
        return target or self.forward_episode.target_description

    @property
    def trajectory(self):
        return self.teacher_trajectory

    @property
    def teacher_actions(self):
        # The released HETT teacher rollout follows teacher_trajectory directly;
        # reverse action ids are deliberately not fabricated.
        return []

    @property
    def time_step(self):
        return len(self.teacher_trajectory)

    @property
    def target_description(self):
        start = self.start_pose.xy
        goal = self.target_position.xy
        dx, dy = goal.x - start.x, goal.y - start.y
        distance_m = math.hypot(dx, dy)
        direction = metric_direction_name(dx, dy)
        return (
            f'Fly {distance_m:.0f} meters {direction} to reach the route '
            'starting point.'
        )

    @property
    def target_processed_description(self):
        return self.forward_episode.target_processed_description

    @property
    def target_type(self):
        return self.forward_episode.target_type

    @property
    def initial_distance_to_target(self):
        return max(self.start_pose.xy.dist_to(self.target_position.xy), 1e-6)
