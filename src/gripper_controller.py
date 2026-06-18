"""
gripper_controller.py
Gripper control for parallel jaw grippers.
"""

import numpy as np
from typing import Optional


class GripperController:
    """
    Controller for parallel jaw gripper.
    """

    def __init__(
        self,
        gripper,
        open_position: np.ndarray,
        closed_position: np.ndarray,
        max_force: float = 100.0,
        force_threshold: float = 5.0,
    ):
        self.gripper = gripper
        self.open_position = open_position
        self.closed_position = closed_position
        self.max_force = max_force
        self.force_threshold = force_threshold

        self.current_width = np.linalg.norm(open_position)
        self.target_width = self.current_width
        self.is_moving = False

    def open(self, width: Optional[float] = None) -> bool:
        if width is None:
            width = np.linalg.norm(self.open_position)

        self.target_width = width
        self.is_moving = True

        ratio = width / np.linalg.norm(self.open_position)
        target_positions = self.open_position * ratio

        self.gripper.apply_action(target_positions)

        current_positions = self.gripper.get_joint_positions()
        error = np.linalg.norm(current_positions - target_positions)

        if error < 0.005:
            self.is_moving = False
            self.current_width = width
            return True

        return False

    def close(self, width: Optional[float] = None, force_control: bool = True) -> bool:
        if width is None:
            width = 0.0

        self.target_width = width
        self.is_moving = True

        ratio = width / np.linalg.norm(self.open_position) if width > 0 else 0
        target_positions = self.open_position * ratio

        self.gripper.apply_action(target_positions)

        if force_control:
            force = self.get_applied_force()
            if force > self.force_threshold:
                self.is_moving = False
                self.current_width = width
                return True

        current_positions = self.gripper.get_joint_positions()
        error = np.linalg.norm(current_positions - target_positions)

        if error < 0.005:
            self.is_moving = False
            self.current_width = width
            return True

        return False

    def get_applied_force(self) -> float:
        efforts = self.gripper.get_joint_efforts()
        if efforts is None or len(efforts) == 0:
            return 0.0
        return np.sum(np.abs(efforts))

    def get_joint_positions(self) -> np.ndarray:
        return self.gripper.get_joint_positions()

    def reset(self):
        self.open()
        self.is_moving = False
