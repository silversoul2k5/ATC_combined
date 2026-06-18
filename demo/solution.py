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
    from omni.isaac.core.utils.xforms import get_world_pose
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

class AlgSolution:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state = "INIT"
        self.step_idx = 0

        # Piper Joint info (8 joints total: 6 arm + 2 gripper)
        # Default home position from TaskEEnvPiperCfg
        self.default_joint_pos = torch.tensor(
            [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035],
            device=self.device
        )

        # Action scale from config
        self.action_scale = 0.5
        self.gripper_open_pos = 0.035
        self.gripper_close_pos = -0.015

        # Task state
        self.object_names = ["Object1", "Object2", "Object3"]
        self.target_object = None
        self.objects_remaining = []

        # Robot base position in Task E
        self.base_pos = torch.tensor([1.4, 0.0, 0.8266], device=self.device)

        self.initialized = False
        self.debug_first = True

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {
            "arm": {
                "mode": "position",
                "scale": self.action_scale,
                "clip": None,
            }
        }

    def _detect_objects(self):
        """Ground truth object detection using USD stage."""
        if not HAS_ISAAC:
            return []
        try:
            detections = []
            for name in self.object_names:
                prim_path = f"/World/envs/env_0/{name}"
                pos, rot = get_world_pose(prim_path)
                detections.append({
                    "name": name,
                    "pos": torch.tensor(pos, device=self.device, dtype=torch.float32),
                    "rot": torch.tensor(rot, device=self.device, dtype=torch.float32)
                })
            return detections
        except Exception as e:
            print(f"[AlgSolution] Detection error: {e}")
            return []

    def _heuristic_reach(self, current_qpos, target_world_pos, gripper_width):
        """Simple joint-space reach heuristic for Piper robot."""
        # target_world_pos is [x, y, z]
        rel_pos = target_world_pos - self.base_pos

        target_arm_qpos = current_qpos[:6].clone()

        # Joint 1: Base rotation
        target_arm_qpos[0] = torch.atan2(rel_pos[1], rel_pos[0] - 0.1) # Offset for base center

        # Joint 2, 3: Simplified reach
        dist = torch.norm(rel_pos[:2])
        target_arm_qpos[1] = 1.2 - dist * 0.4
        target_arm_qpos[2] = -1.5 + (target_world_pos[2] - 0.8) * 2.0

        return self._build_action(target_arm_qpos, gripper_width)

    def _build_action(self, arm_qpos, gripper_width):
        """Format action: [j1...j6, g1, g2] scaled."""
        target_qpos = torch.zeros(8, device=self.device)
        target_qpos[:6] = arm_qpos[:6]
        target_qpos[6] = gripper_width
        target_qpos[7] = -gripper_width

        # Action = (target - default) / scale
        action = (target_qpos - self.default_joint_pos) / self.action_scale
        return action.cpu().numpy().tolist()

    def predicts(self, obs, current_score):
        if self.debug_first:
            print(f"[AlgSolution] obs keys: {obs.keys()}")
            print(f"[AlgSolution] proprio shape: {obs['proprio'].shape}")
            self.debug_first = False

        if not self.initialized:
            self.objects_remaining = self._detect_objects()
            print(f"[AlgSolution] Detected {len(self.objects_remaining)} objects.")
            self.initialized = True
            self.state = "SELECT_TARGET"

        # Parse proprio: [8 pos, 8 vel, 8 action]
        proprio = obs['proprio'].to(self.device)
        qpos_rel = proprio[0, :8]
        current_qpos = qpos_rel + self.default_joint_pos

        if self.state == "SELECT_TARGET":
            if not self.objects_remaining:
                self.state = "DONE"
                return {"action": [0.0] * 8, "giveup": True}
            else:
                self.target_object = self.objects_remaining.pop(0)
                self.state = "APPROACH"
                self.step_idx = 0

        # State Machine Logic
        if self.state == "APPROACH":
            self.step_idx += 1
            target_pos = self.target_object["pos"].clone()
            target_pos[2] += 0.2
            action = self._heuristic_reach(current_qpos, target_pos, self.gripper_open_pos)
            if self.step_idx > 60:
                self.state = "DESCEND"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "DESCEND":
            self.step_idx += 1
            target_pos = self.target_object["pos"].clone()
            target_pos[2] += 0.05
            action = self._heuristic_reach(current_qpos, target_pos, self.gripper_open_pos)
            if self.step_idx > 40:
                self.state = "GRASP"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "GRASP":
            self.step_idx += 1
            target_pos = self.target_object["pos"].clone()
            target_pos[2] += 0.05
            action = self._heuristic_reach(current_qpos, target_pos, self.gripper_close_pos)
            if self.step_idx > 20:
                self.state = "LIFT"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "LIFT":
            self.step_idx += 1
            target_pos = self.target_object["pos"].clone()
            target_pos[2] += 0.35
            action = self._heuristic_reach(current_qpos, target_pos, self.gripper_close_pos)
            if self.step_idx > 60:
                self.state = "MOVE_TO_BASKET"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "MOVE_TO_BASKET":
            self.step_idx += 1
            # Basket position in local coordinates is around [1.08, -0.3, 0.97]
            basket_pos = torch.tensor([1.1, -0.3, 1.1], device=self.device)
            action = self._heuristic_reach(current_qpos, basket_pos, self.gripper_close_pos)
            if self.step_idx > 100:
                self.state = "PLACE"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "PLACE":
            self.step_idx += 1
            basket_pos = torch.tensor([1.1, -0.3, 0.95], device=self.device)
            action = self._heuristic_reach(current_qpos, basket_pos, self.gripper_open_pos)
            if self.step_idx > 40:
                self.state = "SELECT_TARGET"
                self.step_idx = 0
            return {"action": action, "giveup": False}

        elif self.state == "DONE":
            return {"action": [0.0] * 8, "giveup": True}

        return {"action": [0.0] * 8, "giveup": False}
