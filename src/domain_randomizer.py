"""
domain_randomizer.py
Domain randomization for sim-to-real transfer.
"""

import numpy as np
import random
from typing import Dict

from omni.isaac.core.utils.prims import get_prim_at_path


class DomainRandomizer:
    def __init__(self, environment, config: Dict):
        self.environment = environment
        self.config = config
        self.light_intensity_range = config.get('light_intensity', [1000, 8000])
        self.light_color_variation = config.get('light_color_var', 0.1)
        self.object_color_variation = config.get('object_color_var', 0.3)
        self.mass_variation = config.get('mass_var', 0.2)
        self.friction_variation = config.get('friction_var', 0.3)
        self.camera_pos_variation = config.get('camera_pos_var', 0.05)

    def randomize_all(self):
        self.randomize_lighting()
        self.randomize_object_appearance()
        self.randomize_object_physics()
        self.randomize_camera()
        self.randomize_table()

    def randomize_lighting(self):
        main_intensity = random.uniform(*self.light_intensity_range)
        main_prim = get_prim_at_path("/World/lighting/main_light")
        if main_prim:
            main_prim.GetAttribute("intensity").Set(main_intensity)

        fill_intensity = random.uniform(
            self.light_intensity_range[0] * 0.3,
            self.light_intensity_range[1] * 0.5,
        )
        fill_prim = get_prim_at_path("/World/lighting/fill_light")
        if fill_prim:
            fill_prim.GetAttribute("intensity").Set(fill_intensity)

        dome_intensity = random.uniform(100, 600)
        dome_prim = get_prim_at_path("/World/lighting/dome")
        if dome_prim:
            dome_prim.GetAttribute("intensity").Set(dome_intensity)

    def randomize_object_appearance(self):
        for obj_info in self.environment.objects:
            obj_prim = obj_info['prim']
            color = np.random.uniform(0.2, 1.0, 3)
            obj_prim.set_color(color)

    def randomize_object_physics(self):
        for obj_info in self.environment.objects:
            obj_prim = obj_info['prim']
            base_mass = obj_info.get('mass', 0.1)
            mass = base_mass * random.uniform(1 - self.mass_variation, 1 + self.mass_variation)
            obj_prim.set_mass(mass)

    def randomize_camera(self):
        if self.environment.camera is None:
            return
        current_pos = self.environment.camera.get_world_pose()[0]
        noise = np.random.normal(0, self.camera_pos_variation, 3)
        new_pos = current_pos + noise
        new_pos[2] = max(new_pos[2], 1.0)
        self.environment.camera.set_world_pose(position=new_pos)

    def randomize_table(self):
        wood_colors = [
            [0.3, 0.25, 0.2],
            [0.5, 0.4, 0.3],
            [0.7, 0.6, 0.5],
            [0.4, 0.4, 0.4],
        ]
        color = random.choice(wood_colors)
        self.environment.table.set_color(np.array(color))

    def randomize_object_positions(self, max_attempts: int = 100):
        table_bounds = self.environment.table_bounds
        table_z = self.environment.table_surface_z

        positions = []
        for obj_info in self.environment.objects:
            obj_size = obj_info['size']
            radius = max(obj_size[:2]) / 2 + 0.02

            for _ in range(max_attempts):
                x = random.uniform(table_bounds['x_min'], table_bounds['x_max'])
                y = random.uniform(table_bounds['y_min'], table_bounds['y_max'])
                z = table_z + obj_size[-1] / 2 + 0.01

                overlap = False
                for px, py, pr in positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < (radius + pr):
                        overlap = True
                        break

                if not overlap:
                    positions.append((x, y, radius))
                    obj_prim = obj_info['prim']
                    obj_prim.set_world_pose(position=np.array([x, y, z]))
                    obj_prim.set_linear_velocity(np.zeros(3))
                    obj_prim.set_angular_velocity(np.zeros(3))
                    break
