"""
motion_controller.py
Motion planning and control for ATEC2026 Table Clean-up.
Handles IK, trajectory execution, and collision avoidance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import CubicSpline

from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.motion_generation import (
    ArticulationMotionPolicy,
    RmpFlow,
    interface_config_loader,
)
from omni.isaac.core.articulations import Articulation


class MotionController:
    """
    Motion controller for robot arm manipulation.
    Uses RMPflow for reactive motion planning with collision avoidance.
    """

    def __init__(
        self,
        robot: Articulation,
        end_effector_frame_name: str,
        robot_description_path: str,
        rmp_config_path: str,
        robot_usd_path: str,
    ):
        self.robot = robot
        self.end_effector_frame_name = end_effector_frame_name
        self.num_joints = len(robot.dof_names)
        self.joint_names = robot.dof_names
        self.joint_limits = self._get_joint_limits()

        # Initialize RMPflow motion policy
        self.rmpflow = RmpFlow(
            robot_description_path=robot_description_path,
            urdf_path=robot_usd_path,
            rmpflow_config_path=rmp_config_path,
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=0.00334,
        )

        self.motion_policy = ArticulationMotionPolicy(
            robot_articulation=robot,
            motion_policy=self.rmpflow,
        )

        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_ee_pose = None
        self.active_trajectory = None
        self.trajectory_time = 0.0
        self.trajectory_duration = 0.0

    def _get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        limits = {}
        for i, name in enumerate(self.joint_names):
            lower = self.robot.dof_properties["lower"][i]
            upper = self.robot.dof_properties["upper"][i]
            limits[name] = (lower, upper)
        return limits

    def update_state(self):
        self.current_joint_positions = self.robot.get_joint_positions()
        self.current_joint_velocities = self.robot.get_joint_velocities()
        self.rmpflow.update_world()

    def move_to_pose(
        self,
        target_pose: np.ndarray,
        duration: float = 2.0,
        tolerance: float = 0.02,
        max_velocity: float = 0.5,
    ) -> bool:
        self.update_state()

        target_position = target_pose[:3, 3]
        target_orientation = self._matrix_to_quat(target_pose[:3, :3])

        self.rmpflow.set_end_effector_target(
            target_position=target_position,
            target_orientation=target_orientation,
        )

        action = self.motion_policy.get_next_articulation_action(
            current_joint_positions=self.current_joint_positions,
        )

        self.robot.apply_action(action)

        current_ee_pos = self.get_end_effector_position()
        error = np.linalg.norm(current_ee_pos - target_position)

        return error < tolerance

    def get_end_effector_pose(self) -> np.ndarray:
        ee_index = self.robot.body_names.index(self.end_effector_frame_name)
        position, orientation = self.robot.get_body_coms()[ee_index]
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = self._quat_to_matrix(orientation)
        return pose

    def get_end_effector_position(self) -> np.ndarray:
        pose = self.get_end_effector_pose()
        return pose[:3, 3]

    def _matrix_to_quat(self, R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])

    def _quat_to_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def add_obstacle(self, obstacle_path: str):
        self.rmpflow.add_obstacle(obstacle_path)

    def clear_obstacles(self):
        self.rmpflow.clear_obstacles()
