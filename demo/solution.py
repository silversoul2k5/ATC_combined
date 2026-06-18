import torch
import numpy as np
from typing import Any, Dict, List, Optional
import sys
import os

# Add project directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    import omni.usd
    from isaaclab.assets import Articulation
    from atec_rl_lab.utils import CartesianController
    from atec_rl_lab.tasks.task_e.env_cfg import TABLE_TOP_Z, BASKET_SUCCESS_CENTER
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False
    TABLE_TOP_Z = 0.8266
    BASKET_SUCCESS_CENTER = (1.08, -0.30, 0.9766)

class AlgSolution:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state = "INIT"
        self.step_idx = 0

        # Robot handles
        self.robot: Optional[Articulation] = None
        self.controller: Optional[CartesianController] = None

        # Task state
        self.object_names = ["Object1", "Object2", "Object3"] # From TaskESceneCfg
        self.target_object_name = None
        self.objects_remaining = []
        self.grasp_plan = None

        # Joint info
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = ["joint7", "joint8"]
        self.default_joint_pos = torch.tensor(
            [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035],
            device=self.device
        )

        # Constants
        self.action_scale = 0.5
        self.gripper_open_pos = 0.035
        self.gripper_close_pos = -0.015

        self.initialized = False

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {
            "arm": {
                "mode": "position",
                "scale": self.action_scale,
                "clip": None,
            }
        }

    def _lazy_init(self):
        if not HAS_ISAAC:
            return

        # Try to find the robot articulation in the stage
        stage = omni.usd.get_context().get_stage()
        # In Task E, robot is at /World/envs/env_0/Robot (if num_envs=1)
        # We can search for the first articulation that matches the Piper name
        from isaaclab.assets import ArticulationCfg

        # Since we are running in play_atec_task.py, the env is already created.
        # We try to get the robot from the stage.
        # This is a bit hacky but common for scripted solutions in Isaac Sim.

        # Find any prim that looks like the robot
        robot_prim_path = None
        for prim in stage.Traverse():
            if prim.GetName() == "Robot" and "env_0" in str(prim.GetPath()):
                robot_prim_path = str(prim.GetPath())
                break

        if robot_prim_path:
            # We don't have the ArticulationCfg here, but we can try to find it via IsaacLab's registry or just re-create it.
            # However, Articulation needs a config.
            # A better way might be to wait until predicts is called and try to find the robot in the scene.
            pass

    def _get_ee_pose(self, pos, rot_quat):
        """Helper to create pose tensor [pos, quat]"""
        return torch.cat([
            torch.tensor(pos, device=self.device).view(1, 3),
            torch.tensor(rot_quat, device=self.device).view(1, 4)
        ], dim=-1)

    def _detect_objects(self):
        """Ground truth object detection using USD stage."""
        if not HAS_ISAAC:
            return []

        stage = omni.usd.get_context().get_stage()
        detections = []
        for name in self.object_names:
            # Task E objects are at /World/envs/env_0/ObjectName
            prim_path = f"/World/envs/env_0/{name}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                from omni.isaac.core.utils.xforms import get_world_pose
                pos, rot = get_world_pose(prim_path)
                # Convert to local env frame if needed.
                # In play_atec_task.py with 1 env, env_origin is usually (0,0,0)
                detections.append({
                    "name": name,
                    "pos": torch.tensor(pos, device=self.device),
                    "rot": torch.tensor(rot, device=self.device)
                })
        return detections

    def predicts(self, obs, current_score):
        if not self.initialized:
            # Initialize robot and controller on first call
            if HAS_ISAAC:
                # We need to find the robot in the scene.
                # Since we don't have the env object, we try to use the stage.
                try:
                    # Look for robot Articulation
                    # This requires knowledge of how IsaacLab spawns things.
                    # Alternatively, we can just use the joint positions from obs
                    # and implement a simple joint-space controller or
                    # use a library like 'lula' or 'pink' if available.

                    # For this solution, we will use a joint-space FSM
                    # since full IK setup without env access is complex.
                    self.objects_remaining = self._detect_objects()
                except Exception as e:
                    print(f"Initialization error: {e}")

            self.initialized = True
            self.state = "INIT"

        # Parse proprio
        # obs['proprio'] is (1, 24) -> [pos(8), vel(8), action(8)]
        proprio = obs['proprio']
        qpos = proprio[0, :8] + self.default_joint_pos[0] # Current joint positions

        action = torch.zeros(8, device=self.device)
        gripper_cmd = self.gripper_open_pos

        if self.state == "INIT":
            self.objects_remaining = self._detect_objects()
            self.state = "SELECT_TARGET"

        if self.state == "SELECT_TARGET":
            if not self.objects_remaining:
                self.state = "DONE"
            else:
                self.target_object = self.objects_remaining.pop(0)
                self.state = "MOVE_TO_PREGRASP"
                self.step_idx = 0

        # Simple joint-space heuristic for Task E
        # Note: This is a placeholder for a more advanced IK-based FSM.
        # In a real competition, you'd use CartesianController.

        target_qpos = self.default_joint_pos[0].clone()

        if self.state == "MOVE_TO_PREGRASP":
            # Move arm above object
            # This is hard to do precisely without IK.
            # We'll use a very simple movement toward the object XY
            obj_pos = self.target_object["pos"]

            # Heuristic mapping for Piper robot
            # joint1 controls horizontal angle
            # joint2, 3 control reach/height
            target_qpos[0] = torch.atan2(obj_pos[1], obj_pos[0] - 1.3)
            target_qpos[1] = 0.8
            target_qpos[2] = -1.0

            gripper_cmd = self.gripper_open_pos
            self.step_idx += 1
            if self.step_idx > 100:
                self.state = "DESCEND"
                self.step_idx = 0

        elif self.state == "DESCEND":
            obj_pos = self.target_object["pos"]
            target_qpos[0] = torch.atan2(obj_pos[1], obj_pos[0] - 1.3)
            target_qpos[1] = 1.2
            target_qpos[2] = -1.5

            gripper_cmd = self.gripper_open_pos
            self.step_idx += 1
            if self.step_idx > 50:
                self.state = "GRASP"
                self.step_idx = 0

        elif self.state == "GRASP":
            obj_pos = self.target_object["pos"]
            target_qpos[0] = torch.atan2(obj_pos[1], obj_pos[0] - 1.3)
            target_qpos[1] = 1.2
            target_qpos[2] = -1.5

            gripper_cmd = self.gripper_close_pos
            self.step_idx += 1
            if self.step_idx > 20:
                self.state = "LIFT"
                self.step_idx = 0

        elif self.state == "LIFT":
            target_qpos[1] = 0.5
            target_qpos[2] = -1.0
            gripper_cmd = self.gripper_close_pos
            self.step_idx += 1
            if self.step_idx > 50:
                self.state = "MOVE_TO_BASKET"
                self.step_idx = 0

        elif self.state == "MOVE_TO_BASKET":
            # Basket at (1.08, -0.3)
            target_qpos[0] = -0.8
            target_qpos[1] = 0.5
            target_qpos[2] = -1.0
            gripper_cmd = self.gripper_close_pos
            self.step_idx += 1
            if self.step_idx > 100:
                self.state = "PLACE"
                self.step_idx = 0

        elif self.state == "PLACE":
            target_qpos[0] = -0.8
            target_qpos[1] = 1.0
            target_qpos[2] = -1.2
            gripper_cmd = self.gripper_open_pos
            self.step_idx += 1
            if self.step_idx > 30:
                self.state = "SELECT_TARGET"
                self.step_idx = 0

        elif self.state == "DONE":
            target_qpos = self.default_joint_pos[0].clone()
            return {"action": [0.0] * 8, "giveup": True}

        # Format action
        target_qpos[6] = gripper_cmd
        target_qpos[7] = -gripper_cmd # Piper gripper is symmetric

        # Action is (target - default) / scale
        action = (target_qpos - self.default_joint_pos[0]) / self.action_scale

        return {"action": action.tolist(), "giveup": False}
