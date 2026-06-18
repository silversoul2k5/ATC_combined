"""
perception.py
Object detection and localization for ATEC2026 Table Clean-up.
"""

import numpy as np
from typing import List, Dict, Optional
import omni.isaac.core.utils.xforms as xform_utils
from omni.isaac.core.utils.prims import get_prim_at_path


class DetectedObject:
    def __init__(self, name, position, orientation, size, confidence=1.0):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.size = size
        self.confidence = confidence
        self.in_basket = False


class ObjectPerception:
    def __init__(self, camera=None, confidence_threshold=0.7):
        self.camera = camera
        self.confidence_threshold = confidence_threshold
        self.object_names = ["object_1", "object_2", "object_3"]

    def detect_objects(self) -> List[DetectedObject]:
        """
        Detect objects in the scene.
        Uses ground truth from simulation for the modular baseline.
        """
        detections = []
        for name in self.object_names:
            prim_path = f"/World/objects/{name}"
            prim = get_prim_at_path(prim_path)
            if not prim.IsValid():
                continue

            position, orientation = xform_utils.get_world_pose(prim_path)

            # In a real system, we'd get the size from detection or a database
            # Here we assume some default size or get it from the prim
            scale = prim.GetAttribute("xformOp:scale").Get()
            size = np.array(scale) if scale else np.array([0.05, 0.05, 0.05])

            detections.append(DetectedObject(
                name=name,
                position=position,
                orientation=orientation,
                size=size,
                confidence=1.0
            ))

        return detections

    def filter_detections(self, detections: List[DetectedObject], table_bounds: Dict) -> List[DetectedObject]:
        filtered = []
        for det in detections:
            x, y, z = det.position
            if (table_bounds['x_min'] <= x <= table_bounds['x_max'] and
                table_bounds['y_min'] <= y <= table_bounds['y_max']):
                filtered.append(det)
        return filtered

    def sort_by_pick_priority(self, detections: List[DetectedObject], robot_base_pos: np.ndarray) -> List[DetectedObject]:
        # Sort by distance to robot base (XY plane)
        return sorted(detections, key=lambda d: np.linalg.norm(d.position[:2] - robot_base_pos[:2]))

    def pixel_to_world(self, u, v, depth, camera_intrinsics, camera_pose):
        """
        Helper to project pixel coordinates to world space.
        (Used if doing real vision-based detection)
        """
        # Intrinsic matrix K
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        # Camera space coordinates
        z_c = depth
        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy
        p_c = np.array([x_c, y_c, z_c, 1.0])

        # World space coordinates
        p_w = camera_pose @ p_c
        return p_w[:3]
