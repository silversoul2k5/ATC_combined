#!/usr/bin/env python3
"""
run_simulation.py
Main entry point for ATEC2026 Table Clean-up simulation.
"""

import argparse
import numpy as np
from typing import Dict

from isaacsim.core.api.world import World
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.motion_generation")
enable_extension("omni.isaac.sensor")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from environment import TableCleanupEnvironment
from perception import ObjectPerception
from grasp_planner import GraspPlanner
from motion_controller import MotionController
from pick_place_fsm import PickPlaceFSM
from task_planner import TaskPlanner
from gripper_controller import GripperController
from domain_randomizer import DomainRandomizer
from utils import load_config, Logger, format_time


def setup_simulation(config: Dict):
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.get('physics', {}).get('dt', 1/60),
        rendering_dt=config.get('physics', {}).get('dt', 1/60),
    )
    env = TableCleanupEnvironment(config, seed=42)
    world = env.build_scene(world)
    world.reset()
    return world, env


def initialize_controllers(env, config: Dict):
    perception = ObjectPerception(
        camera=env.camera,
        confidence_threshold=config.get('detection', {}).get('confidence_threshold', 0.7),
    )

    grasp_planner = GraspPlanner(
        approach_height=config.get('grasp', {}).get('approach_height', 0.15),
        grasp_offset=config.get('grasp', {}).get('grasp_offset', 0.02),
        lift_height=config.get('place', {}).get('approach_height', 0.20),
    )

    robot_config = config.get('robot', {})
    motion_controller = MotionController(
        robot=env.robot,
        end_effector_frame_name=robot_config.get('end_effector', {}).get('prim_path', '/World/robot/panda_hand'),
        robot_description_path="/Isaac/Robots/Franka/franka_description.urdf",
        rmp_config_path="/Isaac/Robots/Franka/rmpflow_config.yaml",
        robot_usd_path="/Isaac/Robots/Franka/franka_alt_fingers.usd",
    )

    gripper_config = robot_config.get('gripper', {})
    gripper_controller = GripperController(
        gripper=env.gripper,
        open_position=np.array(gripper_config.get('open_position', [0.04, 0.04])),
        closed_position=np.array(gripper_config.get('closed_position', [0.0, 0.0])),
        max_force=gripper_config.get('max_force', 100.0),
    )

    pick_place_fsm = PickPlaceFSM(
        motion_controller=motion_controller,
        gripper_controller=gripper_controller,
        perception=perception,
        config=config,
    )

    task_planner = TaskPlanner(
        environment=env,
        perception=perception,
        grasp_planner=grasp_planner,
        pick_place_fsm=pick_place_fsm,
        config=config,
    )

    return perception, grasp_planner, motion_controller, gripper_controller, pick_place_fsm, task_planner


def run_episode(world, env, task_planner, pick_place_fsm, domain_randomizer, config, episode_num):
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}")
    print(f"{'='*60}")

    env.reset_objects()
    world.reset()

    if config.get('domain_randomization', {}).get('enabled', False):
        domain_randomizer.randomize_all()
        domain_randomizer.randomize_object_positions()

    task_planner.start_episode(world.current_time)

    max_steps = config.get('episode', {}).get('max_steps', 1000)
    dt = config.get('physics', {}).get('dt', 1/60)

    step = 0
    done = False

    while not done and step < max_steps:
        world.step(render=True)
        action = task_planner.get_next_action(world.current_time)

        if action is None:
            done = True
        elif action['type'] == 'continue_fsm':
            pick_place_fsm.update(dt, world.current_time)
        elif action['type'] == 'pick_place':
            pass

        if step % 100 == 0:
            state = task_planner.episode_state
            print(f"Step {step}: {state.objects_picked}/{state.objects_total} picked")

        step += 1

    result = task_planner.get_episode_result()
    result['episode_num'] = episode_num
    result['total_steps'] = step

    print(f"\nEpisode {episode_num} complete!")
    print(f"  Success rate: {result['success_rate']*100:.1f}%")
    print(f"  Objects picked: {result['objects_picked']}/{result['objects_total']}")
    print(f"  Time: {format_time(result['episode_time'])}")
    print(f"  Success: {result['success']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ATEC2026 Table Clean-up Simulation")
    parser.add_argument("--config", type=str, default="config/scene_config.yaml")
    parser.add_argument("--robot_config", type=str, default="config/robot_config.yaml")
    parser.add_argument("--task_config", type=str, default="config/task_config.yaml")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--domain_randomization", action="store_true")

    args = parser.parse_args()

    scene_config = load_config(args.config)
    robot_config = load_config(args.robot_config)
    task_config = load_config(args.task_config)

    config = {**scene_config.get('scene', {}), **robot_config.get('robot', {}), **task_config.get('task', {})}
    config['domain_randomization'] = {'enabled': args.domain_randomization}

    print(f"Objects: {config.get('num_objects', 5)}")
    print(f"Robot: {config.get('type', 'franka')}")
    print(f"Episodes: {args.num_episodes}")

    world, env = setup_simulation(scene_config)
    perception, grasp_planner, motion_controller, gripper_controller, pick_place_fsm, task_planner = \
        initialize_controllers(env, config)

    domain_randomizer = DomainRandomizer(env, config)
    logger = Logger("episode_results.txt")

    for episode in range(args.num_episodes):
        result = run_episode(world, env, task_planner, pick_place_fsm, domain_randomizer, config, episode + 1)
        logger.log_episode(result)

    stats = logger.get_statistics()
    print(f"\n{'='*60}")
    print("Final Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Mean success rate: {stats['mean_success_rate']*100:.1f}%")
    print(f"  Successful episodes: {stats['success_count']}/{stats['total_episodes']}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
