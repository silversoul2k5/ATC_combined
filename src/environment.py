"""
environment.py
Scene setup and management for ATEC2026 Table Clean-up.
"""

import numpy as np
from typing import Dict, List, Optional
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import Camera


class TableCleanupEnvironment:
    def __init__(self, config: Dict, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)

        self.objects = []
        self.table = None
        self.basket = None
        self.robot = None
        self.gripper = None
        self.camera = None

        # Extracted from config or defaults
        self.table_center = config.get('table_center', [1.0, 0.0, 0.0])
        self.table_size = config.get('table_size', [0.65, 0.91, 0.66])
        self.table_surface_z = self.table_center[2] + self.table_size[2] / 2

        self.basket_position = config.get('basket_position', [1.08, -0.30, 0.74])
        self.basket_size = config.get('basket_size', [0.4, 0.3, 0.2])

        self.table_bounds = {
            'x_min': self.table_center[0] - self.table_size[0] / 2 + 0.05,
            'x_max': self.table_center[0] + self.table_size[0] / 2 - 0.05,
            'y_min': self.table_center[1] - self.table_size[1] / 2 + 0.05,
            'y_max': self.table_center[1] + self.table_size[1] / 2 - 0.05,
        }

        self.robot_base_position = np.array([1.3, 0.0, self.table_surface_z])

    def build_scene(self, world):
        # Create Table
        self.table = VisualCuboid(
            prim_path="/World/table",
            name="table",
            position=np.array(self.table_center),
            size=1.0,
            scale=np.array(self.table_size),
            color=np.array([0.4, 0.4, 0.4])
        )

        # Create Objects
        obj_types = ["sugar", "mustard", "banana"]
        obj_sizes = [[0.05, 0.05, 0.1], [0.04, 0.04, 0.12], [0.15, 0.03, 0.03]]

        for i in range(self.config.get('num_objects', 3)):
            name = f"object_{i+1}"
            idx = i % len(obj_types)
            size = obj_sizes[idx]

            # Initial position (random on table)
            pos = self._get_random_table_pos(size)

            obj_prim = DynamicCuboid(
                prim_path=f"/World/objects/{name}",
                name=name,
                position=pos,
                size=1.0,
                scale=np.array(size),
                color=np.random.uniform(0.3, 0.9, 3),
                mass=0.1
            )

            self.objects.append({
                'name': name,
                'type': obj_types[idx],
                'size': size,
                'initial_position': pos,
                'prim': obj_prim
            })

        # Create Basket
        self.basket = VisualCuboid(
            prim_path="/World/basket",
            name="basket",
            position=np.array(self.basket_position),
            size=1.0,
            scale=np.array(self.basket_size),
            color=np.array([0.1, 0.1, 0.5])
        )

        # Load Robot
        from omni.isaac.core.articulations import Articulation
        # Using a default Franka if none specified for generic simulation
        robot_type = self.config.get('type', 'franka')
        if robot_type == 'franka':
            from omni.isaac.franka import Franka
            self.robot = world.scene.add(Franka(prim_path="/World/robot", name="robot", position=self.robot_base_position))
            self.gripper = self.robot.gripper
        else:
            # Fallback or custom robot loading logic
            pass

        # Create Camera
        self.camera = Camera(
            prim_path="/World/camera",
            name="top_camera",
            position=np.array([1.0, 0.0, 2.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=np.array([0, 1, 0, 0]) # Looking down
        )

        return world

    def _get_random_table_pos(self, obj_size):
        x = np.random.uniform(self.table_bounds['x_min'], self.table_bounds['x_max'])
        y = np.random.uniform(self.table_bounds['y_min'], self.table_bounds['y_max'])
        z = self.table_surface_z + obj_size[2] / 2 + 0.01
        return np.array([x, y, z])

    def reset_objects(self):
        for obj in self.objects:
            pos = self._get_random_table_pos(obj['size'])
            obj['prim'].set_world_pose(position=pos, orientation=np.array([1, 0, 0, 0]))
            obj['prim'].set_linear_velocity(np.zeros(3))
            obj['prim'].set_angular_velocity(np.zeros(3))
