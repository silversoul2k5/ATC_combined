"""
grasp_planner.py
Grasp planning for ATEC2026 Table Clean-up.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class GraspPlan:
    def __init__(self, pre_grasp_pose, grasp_pose, lift_pose, pre_place_pose, place_pose, post_place_pose, gripper_close_width):
        self.pre_grasp_pose = pre_grasp_pose
        self.grasp_pose = grasp_pose
        self.lift_pose = lift_pose
        self.pre_place_pose = pre_place_pose
        self.place_pose = place_pose
        self.post_place_pose = post_place_pose
        self.gripper_close_width = gripper_close_width


class GraspPlanner:
    def __init__(self, approach_height=0.15, grasp_offset=0.02, lift_height=0.20):
        self.approach_height = approach_height
        self.grasp_offset = grasp_offset
        self.lift_height = lift_height

    def plan_grasp(self, target_object, object_size, basket_position, basket_size, strategy="top_down") -> GraspPlan:
        if strategy == "auto":
            strategy = "top_down"
        obj_pos = target_object.position
        obj_rot = target_object.orientation

        # 1. Grasp Pose (Top-down)
        grasp_pos = obj_pos.copy()
        # Adjust Z to be slightly above the table surface based on object height
        grasp_pos[2] = obj_pos[2] + self.grasp_offset

        # Orientation: Gripper pointing down
        # For Franka/Piper, this is usually a rotation of 180 deg around Y or similar
        # Let's use a standard top-down orientation
        grasp_rot = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()

        grasp_pose = self._make_pose(grasp_pos, grasp_rot)

        # 2. Pre-Grasp Pose
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += self.approach_height
        pre_grasp_pose = self._make_pose(pre_grasp_pos, grasp_rot)

        # 3. Lift Pose
        lift_pos = grasp_pos.copy()
        lift_pos[2] += self.lift_height
        lift_pose = self._make_pose(lift_pos, grasp_rot)

        # 4. Place Pose (In basket)
        place_pos = np.array(basket_position).copy()
        place_pos[2] += 0.1 # Some height above basket bottom
        place_pose = self._make_pose(place_pos, grasp_rot)

        # 5. Pre-Place Pose
        pre_place_pos = place_pos.copy()
        pre_place_pos[2] += self.lift_height
        pre_place_pose = self._make_pose(pre_place_pos, grasp_rot)

        # 6. Post-Place Pose (Retract)
        post_place_pos = pre_place_pos.copy()
        post_place_pose = self._make_pose(post_place_pos, grasp_rot)

        # Gripper width
        gripper_close_width = max(0.01, min(object_size[0], object_size[1]) - 0.01)

        return GraspPlan(
            pre_grasp_pose=pre_grasp_pose,
            grasp_pose=grasp_pose,
            lift_pose=lift_pose,
            pre_place_pose=pre_place_pose,
            place_pose=place_pose,
            post_place_pose=post_place_pose,
            gripper_close_width=gripper_close_width
        )

    def _make_pose(self, pos, rot_mat):
        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
