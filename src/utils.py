"""
utils.py
Utility functions for ATEC2026 Table Clean-up.
"""

import numpy as np
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def is_pose_reached(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    pos_tolerance: float = 0.01,
    rot_tolerance: float = 0.05,
) -> bool:
    pos_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
    R_current = current_pose[:3, :3]
    R_target = target_pose[:3, :3]
    R_diff = R_current.T @ R_target
    trace = np.trace(R_diff)
    rot_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return pos_error < pos_tolerance and rot_error < rot_tolerance


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


class Logger:
    def __init__(self, log_file: str = "episode_log.txt"):
        self.log_file = log_file
        self.episodes = []

    def log_episode(self, result: Dict):
        self.episodes.append(result)
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {len(self.episodes)}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def get_statistics(self) -> Dict:
        if not self.episodes:
            return {}
        success_rates = [ep['success_rate'] for ep in self.episodes]
        times = [ep['episode_time'] for ep in self.episodes]
        return {
            'total_episodes': len(self.episodes),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_time': np.mean(times),
            'success_count': sum(1 for ep in self.episodes if ep.get('success', False)),
        }
