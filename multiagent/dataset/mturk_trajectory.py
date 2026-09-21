import copy
import json
from dataclasses import dataclass
from typing import Literal, Optional

from multiagent.space import Point3D, Pose4D, Pose5D, modulo_radians
from multiagent.defaultpaths import MTURK_TRAJECTORY_DIR
from multiagent.trajectory import straight_line_trajectory
from multiagent.mapdata import GROUND_LEVEL
from typing import List, Dict, Callable

MturkSplit = Literal['train_seen', 'val_seen', 'val_unseen', 'test_unseen']
MturkDifficulty = Literal['easy', 'medium', 'hard', 'all']


def load_mturk_trajectories(split: MturkSplit, difficulty: MturkDifficulty, fix_altitude: Optional[float] = None, trajectory_dir=MTURK_TRAJECTORY_DIR):
    
    difficulty = '' if difficulty == 'all' else  '_' + difficulty
    path = trajectory_dir/f"citynav_{split}{difficulty}.json"
    
    with open(path) as f:
        # print(len(json.load(f)))
        trajectories =  [MTurkTrajectory(**trajectory) for trajectory in json.load(f)]
        
    if fix_altitude:
        trajectories =  [t.fix_altitude(fix_altitude) for t in trajectories]
    
    return trajectories



@dataclass
class MTurkTrajectory:
    area: str
    block: str
    object_ids: List[int]
    ann_ids: List[int]
    descriptions: List[str]
    trajectory: List[Pose5D]
    marker_positions: List[Point3D]
    target_positions: List[Point3D]
    total_score: float
    dist_marker_to_target: float
    split: str
    dist_start_to_target: float = None

    def __post_init__(self):
        if len(self.trajectory[0]) == 5:
            self.trajectory = [Pose5D(x, y, z, yaw, pitch) for x, y, z, yaw, pitch in self.trajectory]
        if len(self.trajectory[0]) == 6:
            self.trajectory = [Pose5D.from_direction_vector(x, y, z, dx, dy, dz) for x, y, z, dx, dy, dz in self.trajectory]
        
        self.marker_positions = [Point3D(x, y, z) for x, y, z in self.marker_positions]
        self.target_positions = [Point3D(x, y, z) for x, y, z in self.target_positions]

        if self.dist_start_to_target is None:
            self.dist_start_to_target = self.start_pose.xyz.dist_to(self.target_position)
    
    @property
    def map_name(self):
        return f"{self.area}_block_{self.block}"

    @property
    def object_id(self):
        return self.object_ids[0]
    
    @property
    def start_pose(self):
        return self.trajectory[0].xyzyaw
    
    @property
    def target_position(self):
        return self.target_positions[-1]

    @property
    def desc_id(self):
        return self.ann_ids[0]
    
    @property
    def trajectory_xyz(self):
        return [pose.xyz for pose in self.trajectory]

    @property
    def extended_trajectory(self):
        return self.trajectory_xyz + [self.marker_positions[-1]]
    
    @property
    def interpolated_trajectory(self):
        interploated_trajectory = [self.start_pose.xyz]
        for src, dst in zip(self.extended_trajectory[:-1], self.extended_trajectory[1:]):
            for pos in straight_line_trajectory(src, dst):
                if interploated_trajectory[-1].dist_to(pos) > 5.:
                    interploated_trajectory.append(pos)
        
        return interploated_trajectory

    @property
    def interpolated_pose_trajectory(self):
        """Five-metre interpolation that retains the human camera yaw."""
        marker = self.marker_positions[-1]
        marker_pose = Pose4D(marker.x, marker.y, marker.z, self.trajectory[-1].yaw)
        source = [pose.xyzyaw for pose in self.trajectory] + [marker_pose]
        interpolated = [source[0]]
        for src, dst in zip(source[:-1], source[1:]):
            segment = straight_line_trajectory(src.xyz, dst.xyz)
            segment_length = max(src.xyz.dist_to(dst.xyz), 1e-6)
            yaw_delta = modulo_radians(dst.yaw - src.yaw)
            for position in segment:
                if interpolated[-1].xyz.dist_to(position) > 5.:
                    alpha = min(1.0, src.xyz.dist_to(position) / segment_length)
                    yaw = modulo_radians(src.yaw + alpha * yaw_delta)
                    interpolated.append(Pose4D(position.x, position.y, position.z, yaw))
        return interpolated
    
    def fix_altitude(self, altitude_from_ground: float):
        
        new_trajectory = copy.deepcopy(self)

        ground_level = GROUND_LEVEL[self.map_name]
        new_z = ground_level + altitude_from_ground
        
        new_trajectory.trajectory = [Pose5D(x, y, new_z, yaw, pitch) for x, y, z, yaw, pitch in self.trajectory]
        new_trajectory.marker_positions = [Point3D(x, y , new_z) for x, y, z in self.marker_positions]
        new_trajectory.dist_start_to_target = new_trajectory.start_pose.xy.dist_to(self.target_position.xy)

        return new_trajectory
