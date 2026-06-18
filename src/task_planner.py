"""
task_planner.py
High-level task planner for ATEC2026 Table Clean-up.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EpisodeState:
    objects_total: int
    objects_picked: int
    objects_placed: int
    objects_remaining: int
    objects_in_basket: int
    episode_time: float
    episode_steps: int
    success_rate: float
    current_target: Optional[str] = None


class TaskPlanner:
    """
    Task planner for multi-object pick-and-place.
    """

    def __init__(
        self,
        environment,
        perception,
        grasp_planner,
        pick_place_fsm,
        config: Dict,
    ):
        self.environment = environment
        self.perception = perception
        self.grasp_planner = grasp_planner
        self.pick_place_fsm = pick_place_fsm
        self.config = config

        self.episode_state = EpisodeState(
            objects_total=0,
            objects_picked=0,
            objects_placed=0,
            objects_remaining=0,
            objects_in_basket=0,
            episode_time=0.0,
            episode_steps=0,
            success_rate=0.0,
        )

        self.object_status = {}
        self.pick_queue = []
        self.failed_objects = []

        self.start_time = 0.0
        self.max_episode_time = config.get('episode', {}).get('max_time', 120.0)
        self.max_episode_steps = config.get('episode', {}).get('max_steps', 1000)

    def start_episode(self, current_time: float):
        self.start_time = current_time
        self.episode_state = EpisodeState(
            objects_total=len(self.environment.objects),
            objects_picked=0,
            objects_placed=0,
            objects_remaining=len(self.environment.objects),
            objects_in_basket=0,
            episode_time=0.0,
            episode_steps=0,
            success_rate=0.0,
        )

        self.object_status = {}
        for obj in self.environment.objects:
            self.object_status[obj['name']] = {
                'detected': True,
                'picked': False,
                'placed': False,
                'in_basket': False,
                'position': obj['initial_position'],
                'retries': 0,
            }

        self._build_pick_queue()
        self.failed_objects = []

        print(f"Episode started: {self.episode_state.objects_total} objects to clean up")

    def _build_pick_queue(self):
        detections = self.perception.detect_objects()
        detections = self.perception.filter_detections(
            detections,
            table_bounds=self.environment.table_bounds,
        )
        sorted_detections = self.perception.sort_by_pick_priority(
            detections,
            self.environment.robot_base_position,
        )
        self.pick_queue = sorted_detections

        if self.pick_queue:
            self.episode_state.current_target = self.pick_queue[0].name

    def get_next_action(self, current_time: float) -> Optional[Dict]:
        self.episode_state.episode_time = current_time - self.start_time
        self.episode_state.episode_steps += 1

        if self._is_episode_complete():
            return None

        if not self.pick_place_fsm.is_complete() and not self.pick_place_fsm.is_failed():
            return {'type': 'continue_fsm'}

        if self.pick_place_fsm.is_complete():
            self._update_object_status(success=True)
            self.pick_place_fsm.reset()

        if self.pick_place_fsm.is_failed():
            self._update_object_status(success=False)
            self.pick_place_fsm.reset()

        if not self.pick_queue:
            self._build_pick_queue()
            if not self.pick_queue:
                return None

        target = self.pick_queue[0]

        if not self._is_valid_target(target):
            self.pick_queue.pop(0)
            return self.get_next_action(current_time)

        obj_size = self._get_object_size(target.name)

        grasp_plan = self.grasp_planner.plan_grasp(
            target,
            obj_size,
            self.environment.basket_position,
            self.environment.basket_size,
            strategy="auto",
        )

        self.pick_place_fsm.start_pick_place(grasp_plan, target)

        return {
            'type': 'pick_place',
            'target': target.name,
            'grasp_plan': grasp_plan,
        }

    def _is_valid_target(self, target) -> bool:
        if target.name not in self.object_status:
            return False
        status = self.object_status[target.name]
        if status['picked'] or status['retries'] >= 3 or status['in_basket']:
            return False
        return True

    def _get_object_size(self, object_name: str) -> Tuple[float, ...]:
        for obj in self.environment.objects:
            if obj['name'] == object_name:
                return obj['size']
        return (0.05, 0.05, 0.05)

    def _update_object_status(self, success: bool):
        target_name = self.episode_state.current_target
        if target_name is None or target_name not in self.object_status:
            return

        status = self.object_status[target_name]

        if success:
            status['picked'] = True
            status['placed'] = True
            status['in_basket'] = True
            self.episode_state.objects_picked += 1
            self.episode_state.objects_placed += 1
            self.episode_state.objects_in_basket += 1
            print(f"Successfully placed {target_name} in basket")
        else:
            status['retries'] += 1
            if status['retries'] >= 3:
                self.failed_objects.append(target_name)

        self.episode_state.objects_remaining = (
            self.episode_state.objects_total - self.episode_state.objects_picked
        )

        if self.pick_queue and self.pick_queue[0].name == target_name:
            self.pick_queue.pop(0)

        if self.episode_state.objects_total > 0:
            self.episode_state.success_rate = (
                self.episode_state.objects_in_basket / self.episode_state.objects_total
            )

    def _is_episode_complete(self) -> bool:
        if self.episode_state.episode_time >= self.max_episode_time:
            return True
        if self.episode_state.episode_steps >= self.max_episode_steps:
            return True
        if self.episode_state.objects_remaining == 0:
            return True
        remaining_valid = sum(
            1 for name, status in self.object_status.items()
            if not status['picked'] and status['retries'] < 3
        )
        if remaining_valid == 0:
            return True
        return False

    def get_episode_result(self) -> Dict:
        return {
            'success_rate': self.episode_state.success_rate,
            'objects_total': self.episode_state.objects_total,
            'objects_picked': self.episode_state.objects_picked,
            'objects_placed': self.episode_state.objects_placed,
            'objects_in_basket': self.episode_state.objects_in_basket,
            'objects_failed': len(self.failed_objects),
            'episode_time': self.episode_state.episode_time,
            'episode_steps': self.episode_state.episode_steps,
            'success': self.episode_state.success_rate >= 0.9,
        }
