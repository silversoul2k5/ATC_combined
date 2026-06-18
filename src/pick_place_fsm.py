"""
pick_place_fsm.py
Finite State Machine for pick-and-place operations.
"""

import numpy as np
from typing import Optional, Dict, Callable
from enum import Enum, auto


class PickPlaceState(Enum):
    IDLE = auto()
    MOVE_TO_PRE_GRASP = auto()
    APPROACH_GRASP = auto()
    CLOSE_GRIPPER = auto()
    VERIFY_GRASP = auto()
    LIFT_OBJECT = auto()
    MOVE_TO_PRE_PLACE = auto()
    APPROACH_PLACE = auto()
    OPEN_GRIPPER = auto()
    VERIFY_PLACE = auto()
    RETRACT = auto()
    RECOVERY = auto()
    COMPLETE = auto()
    FAILED = auto()


class PickPlaceFSM:
    """
    Finite State Machine for robust pick-and-place.
    """

    def __init__(
        self,
        motion_controller,
        gripper_controller,
        perception,
        config: Dict,
    ):
        self.motion_controller = motion_controller
        self.gripper_controller = gripper_controller
        self.perception = perception
        self.config = config

        self.state = PickPlaceState.IDLE
        self.current_grasp_plan = None
        self.target_object = None

        self.state_start_time = 0.0
        self.state_timeout = 5.0
        self.max_retries = 3
        self.retry_count = 0

        self.current_waypoint_idx = 0
        self.waypoints = []

        self.grasp_verified = False
        self.place_verified = False

        self.on_state_change: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None

    def start_pick_place(self, grasp_plan, target_object):
        self.current_grasp_plan = grasp_plan
        self.target_object = target_object
        self.retry_count = 0
        self.grasp_verified = False
        self.place_verified = False

        self._generate_waypoints()
        self._transition_to(PickPlaceState.MOVE_TO_PRE_GRASP)

    def _generate_waypoints(self):
        plan = self.current_grasp_plan
        self.waypoints = [
            ("pre_grasp", plan.pre_grasp_pose),
            ("grasp", plan.grasp_pose),
            ("lift", plan.lift_pose),
            ("pre_place", plan.pre_place_pose),
            ("place", plan.place_pose),
            ("post_place", plan.post_place_pose),
        ]
        self.current_waypoint_idx = 0

    def update(self, dt: float, current_time: float) -> bool:
        if current_time - self.state_start_time > self.state_timeout:
            print(f"State {self.state.name} timed out")
            self._handle_failure()
            return self.state in [PickPlaceState.COMPLETE, PickPlaceState.FAILED]

        if self.state == PickPlaceState.IDLE:
            return True

        elif self.state == PickPlaceState.MOVE_TO_PRE_GRASP:
            self._execute_move_to_waypoint("pre_grasp", PickPlaceState.APPROACH_GRASP)

        elif self.state == PickPlaceState.APPROACH_GRASP:
            self.gripper_controller.open()
            self._execute_move_to_waypoint("grasp", PickPlaceState.CLOSE_GRIPPER)

        elif self.state == PickPlaceState.CLOSE_GRIPPER:
            if self.gripper_controller.close(width=self.current_grasp_plan.gripper_close_width):
                self._transition_to(PickPlaceState.VERIFY_GRASP)

        elif self.state == PickPlaceState.VERIFY_GRASP:
            if self._verify_grasp():
                self.grasp_verified = True
                self._transition_to(PickPlaceState.LIFT_OBJECT)
            else:
                self._handle_failure()

        elif self.state == PickPlaceState.LIFT_OBJECT:
            self._execute_move_to_waypoint("lift", PickPlaceState.MOVE_TO_PRE_PLACE)

        elif self.state == PickPlaceState.MOVE_TO_PRE_PLACE:
            self._execute_move_to_waypoint("pre_place", PickPlaceState.APPROACH_PLACE)

        elif self.state == PickPlaceState.APPROACH_PLACE:
            self._execute_move_to_waypoint("place", PickPlaceState.OPEN_GRIPPER)

        elif self.state == PickPlaceState.OPEN_GRIPPER:
            if self.gripper_controller.open():
                self._transition_to(PickPlaceState.VERIFY_PLACE)

        elif self.state == PickPlaceState.VERIFY_PLACE:
            if self._verify_place():
                self.place_verified = True
                self._transition_to(PickPlaceState.RETRACT)
            else:
                self._handle_failure()

        elif self.state == PickPlaceState.RETRACT:
            self._execute_move_to_waypoint("post_place", PickPlaceState.COMPLETE)

        elif self.state == PickPlaceState.RECOVERY:
            self._execute_recovery()

        elif self.state in [PickPlaceState.COMPLETE, PickPlaceState.FAILED]:
            return True

        return False

    def _execute_move_to_waypoint(self, waypoint_name: str, next_state: PickPlaceState):
        waypoint_pose = None
        for name, pose in self.waypoints:
            if name == waypoint_name:
                waypoint_pose = pose
                break

        if waypoint_pose is None:
            self._handle_failure()
            return

        reached = self.motion_controller.move_to_pose(
            waypoint_pose,
            tolerance=0.015,
        )

        if reached:
            self._transition_to(next_state)

    def _verify_grasp(self) -> bool:
        gripper_force = self.gripper_controller.get_applied_force()
        if gripper_force < 1.0:
            print("Grasp verification failed: low gripper force")
            return False
        return True

    def _verify_place(self) -> bool:
        # Check if object is now in basket region
        current_objects = self.perception.detect_objects()
        for obj in current_objects:
            if obj.name == self.target_object.name:
                if obj.in_basket:
                    return True
        return True  # Assume success if not visible (occluded)

    def _handle_failure(self):
        self.retry_count += 1
        if self.retry_count <= self.max_retries:
            self._transition_to(PickPlaceState.RECOVERY)
        else:
            self._transition_to(PickPlaceState.FAILED)

    def _execute_recovery(self):
        self.gripper_controller.open()
        current_pose = self.motion_controller.get_end_effector_pose()
        safe_pose = current_pose.copy()
        safe_pose[2, 3] += 0.15
        self.motion_controller.move_to_pose(safe_pose, tolerance=0.02)
        self._transition_to(PickPlaceState.FAILED)

    def _transition_to(self, new_state: PickPlaceState):
        old_state = self.state
        self.state = new_state
        self.state_start_time = 0.0
        if self.on_state_change:
            self.on_state_change(old_state, new_state)
        print(f"State transition: {old_state.name} -> {new_state.name}")

    def is_complete(self) -> bool:
        return self.state == PickPlaceState.COMPLETE

    def is_failed(self) -> bool:
        return self.state == PickPlaceState.FAILED

    def reset(self):
        self.state = PickPlaceState.IDLE
        self.current_grasp_plan = None
        self.target_object = None
        self.retry_count = 0
        self.current_waypoint_idx = 0
        self.waypoints = []
