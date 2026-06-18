# Wipe table task simulation for TidyBot++
#
# The robot picks up a sponge from the table and wipes a stain.
# Sponge grasping is handled by the physics engine (mujoco_env.py).
#
# Usage:
#   python wipe_task.py                    # Run with viewer
#   python wipe_task.py --no-viewer        # Run headless
#   python wipe_task.py --show-images      # Show camera images
#   python wipe_task.py --collect-data     # Collect demonstration data
#   python wipe_task.py --num-episodes 5   # Number of episodes

import argparse
import math
import multiprocessing as mp
import time
import numpy as np
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from mujoco_env import MujocoEnv, ShmState, ShmImage

# Scene path
WIPE_SCENE = 'models/stanford_tidybot/scene_wipe.xml'

# Table surface height (z) in world frame
TABLE_HEIGHT = 0.40

# Wiping pose parameters
# Sponge is attached 0.025m below pinch point, sponge half-height is 0.0075m
# So pinch point must be 0.0325m above table for sponge bottom to just touch
WIPE_Z_OFFSET = 0.033
WIPE_ARM_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Sponge initial position (world frame)
SPONGE_POS = np.array([0.55, 0.15])

# Wiping area center on table (world frame)
WIPE_CENTER = np.array([0.55, -0.05])

# Wiping area half-sizes
WIPE_HALF_X = 0.08
WIPE_HALF_Y = 0.08

# Motion parameters
NUM_ZIGZAG_PASSES = 4
STEPS_PER_PASS = 20
STEPS_APPROACH = 15
STEPS_RETREAT = 15
STEPS_GRIP = 10
STEPS_RELEASE = 30
STEPS_TRANSITION = 15  # Move from sponge to wipe area

# Gripper positions
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0


def _smooth(t):
    """Smooth ease-in-out interpolation (0->1)."""
    return 0.5 * (1 - math.cos(math.pi * t))


def generate_wipe_trajectory(base_pose, arm_pos_init):
    """Generate a complete wipe task trajectory.

    Phases:
    0. Move above sponge (gripper open)
    1. Lower to sponge
    2. Close gripper (grasp sponge)
    3. Lift sponge up
    4. Move above stain area
    5. Lower to table (wipe height)
    6. Zigzag wipe
    7. Retreat (lift up)
    8. Release sponge
    """
    actions = []
    base_height = 0.335  # gen3/base_link z offset

    # Heights in arm local frame
    safe_z = arm_pos_init[2]  # Safe height (initial)
    sponge_grasp_z = TABLE_HEIGHT + 0.025 - base_height  # Sponge center on table
    wipe_z = TABLE_HEIGHT + WIPE_Z_OFFSET - base_height  # Wiping height

    # Positions in arm local frame
    sponge_local = np.array([
        SPONGE_POS[0] - base_pose[0],
        SPONGE_POS[1] - base_pose[1],
    ])
    wipe_local = np.array([
        WIPE_CENTER[0] - base_pose[0],
        WIPE_CENTER[1] - base_pose[1],
    ])

    # --- Phase 0: Open gripper and stay in place ---
    STEPS_OPEN_GRIPPER = 15
    for i in range(STEPS_OPEN_GRIPPER):
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos_init.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_OPEN]),
        })

    # --- Phase 1: Move above sponge (gripper open) ---
    above_sponge = np.array([sponge_local[0], sponge_local[1], safe_z])
    for i in range(STEPS_APPROACH):
        t = _smooth((i + 1) / STEPS_APPROACH)
        arm_pos = arm_pos_init + t * (above_sponge - arm_pos_init)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_OPEN]),
        })

    # --- Phase 1: Lower to sponge ---
    at_sponge = np.array([sponge_local[0], sponge_local[1], sponge_grasp_z])
    for i in range(STEPS_APPROACH):
        t = _smooth((i + 1) / STEPS_APPROACH)
        arm_pos = above_sponge + t * (at_sponge - above_sponge)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_OPEN]),
        })

    # --- Phase 2: Close gripper (grasp sponge) ---
    for i in range(STEPS_GRIP):
        t = _smooth((i + 1) / STEPS_GRIP)
        gripper = GRIPPER_OPEN + t * (GRIPPER_CLOSED - GRIPPER_OPEN)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': at_sponge.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([gripper]),
        })

    # --- Phase 3: Lift sponge up ---
    above_sponge_grasped = np.array([sponge_local[0], sponge_local[1], safe_z])
    for i in range(STEPS_APPROACH):
        t = _smooth((i + 1) / STEPS_APPROACH)
        arm_pos = at_sponge + t * (above_sponge_grasped - at_sponge)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_CLOSED]),
        })

    # --- Phase 4: Move above wipe area ---
    above_wipe = np.array([wipe_local[0], wipe_local[1], safe_z])
    for i in range(STEPS_TRANSITION):
        t = _smooth((i + 1) / STEPS_TRANSITION)
        arm_pos = above_sponge_grasped + t * (above_wipe - above_sponge_grasped)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_CLOSED]),
        })

    # --- Phase 5: Lower to table (wipe height) ---
    at_wipe = np.array([wipe_local[0], wipe_local[1], wipe_z])
    for i in range(STEPS_APPROACH):
        t = _smooth((i + 1) / STEPS_APPROACH)
        arm_pos = above_wipe + t * (at_wipe - above_wipe)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_CLOSED]),
        })

    # --- Phase 6: Zigzag wiping ---
    y_positions = np.linspace(
        WIPE_CENTER[1] - WIPE_HALF_Y,
        WIPE_CENTER[1] + WIPE_HALF_Y,
        NUM_ZIGZAG_PASSES + 1,
    )

    for pass_idx in range(NUM_ZIGZAG_PASSES):
        y_start = y_positions[pass_idx] - base_pose[1]
        y_end = y_positions[pass_idx + 1] - base_pose[1]

        if pass_idx % 2 == 0:
            x_start = WIPE_CENTER[0] - WIPE_HALF_X - base_pose[0]
            x_end = WIPE_CENTER[0] + WIPE_HALF_X - base_pose[0]
        else:
            x_start = WIPE_CENTER[0] + WIPE_HALF_X - base_pose[0]
            x_end = WIPE_CENTER[0] - WIPE_HALF_X - base_pose[0]

        for i in range(STEPS_PER_PASS):
            t = _smooth((i + 1) / STEPS_PER_PASS)
            x = x_start + t * (x_end - x_start)
            y = y_start + t * (y_end - y_start)
            arm_pos = np.array([x, y, wipe_z])
            actions.append({
                'base_pose': base_pose.copy(),
                'arm_pos': arm_pos.copy(),
                'arm_quat': WIPE_ARM_QUAT.copy(),
                'gripper_pos': np.array([GRIPPER_CLOSED]),
            })

    # --- Phase 7: Retreat (lift arm up) ---
    final_arm_pos = actions[-1]['arm_pos'].copy()
    retreat_arm_pos = np.array([
        final_arm_pos[0], final_arm_pos[1], safe_z
    ])
    for i in range(STEPS_RETREAT):
        t = _smooth((i + 1) / STEPS_RETREAT)
        arm_pos = final_arm_pos + t * (retreat_arm_pos - final_arm_pos)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([GRIPPER_CLOSED]),
        })

    # --- Phase 8: Release sponge ---
    for i in range(STEPS_RELEASE):
        t = _smooth((i + 1) / STEPS_RELEASE)
        gripper = GRIPPER_CLOSED + t * (GRIPPER_OPEN - GRIPPER_CLOSED)
        actions.append({
            'base_pose': base_pose.copy(),
            'arm_pos': retreat_arm_pos.copy(),
            'arm_quat': WIPE_ARM_QUAT.copy(),
            'gripper_pos': np.array([gripper]),
        })

    return actions


class WipeMujocoEnv(MujocoEnv):
    """MujocoEnv subclass that uses the wipe scene."""

    def __init__(self, render_images=True, show_viewer=True,
                 show_images=False):
        self.mjcf_path = WIPE_SCENE
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)

        self.shm_state = ShmState()

        if self.render_images:
            import mujoco
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(
                    ShmImage(camera_name, width, height)
                )

        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            mp.Process(
                target=self.visualizer_loop, daemon=True
            ).start()


def run_wipe_episode(env, collect_data=False, output_dir='data/wipe_demos'):
    """Run a single wipe task episode."""
    env.reset()
    time.sleep(0.5)
    obs = env.get_obs()

    base_pose = obs['base_pose'].copy()
    arm_pos_init = obs['arm_pos'].copy()
    print(f'Initial base pose: {base_pose}')
    print(f'Initial arm pos: {arm_pos_init}')

    actions = generate_wipe_trajectory(base_pose, arm_pos_init)
    print(f'Generated {len(actions)} actions for wipe task')

    writer = None
    if collect_data:
        writer = EpisodeWriter(output_dir)

    # Phase boundaries for logging
    p = 0
    phases = []
    p += 15; phases.append((p, 'OpenGripper'))
    p += STEPS_APPROACH; phases.append((p, 'MoveAboveSponge'))
    p += STEPS_APPROACH; phases.append((p, 'LowerToSponge'))
    p += STEPS_GRIP; phases.append((p, 'GripSponge'))
    p += STEPS_APPROACH; phases.append((p, 'LiftSponge'))
    p += STEPS_TRANSITION; phases.append((p, 'MoveToWipe'))
    p += STEPS_APPROACH; phases.append((p, 'LowerToTable'))
    p += NUM_ZIGZAG_PASSES * STEPS_PER_PASS; phases.append((p, 'Wipe'))
    p += STEPS_RETREAT; phases.append((p, 'Retreat'))
    p += STEPS_RELEASE; phases.append((p, 'Release'))

    def get_phase(step_idx):
        prev_bound = 0
        for bound, name in phases:
            if step_idx < bound:
                return name
            prev_bound = bound
        return 'Done'

    start_time = time.time()
    try:
        for step_idx, action in enumerate(actions):
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            env.step(action)
            obs = env.get_obs()

            if writer is not None:
                writer.step(obs, action)

            if step_idx % 10 == 0:
                phase = get_phase(step_idx)
                print(f'  Step {step_idx}/{len(actions)} [{phase}] '
                      f'arm: {obs["arm_pos"].round(3)} '
                      f'gripper: {obs["gripper_pos"][0]:.2f}')

    except KeyboardInterrupt:
        print('\nEpisode interrupted by user')

    if writer is not None and len(writer) > 0:
        print(f'Saving episode with {len(writer)} steps...')
        writer.flush_async()
        writer.wait_for_flush()

    return obs


def main(args):
    env = WipeMujocoEnv(
        render_images=not args.no_images,
        show_viewer=not args.no_viewer,
        show_images=args.show_images,
    )

    try:
        for ep_idx in range(args.num_episodes):
            print(f'\n=== Episode {ep_idx + 1}/{args.num_episodes} ===')
            run_wipe_episode(
                env,
                collect_data=args.collect_data,
                output_dir=args.output_dir,
            )
            if ep_idx < args.num_episodes - 1:
                print('Waiting 2 seconds before next episode...')
                time.sleep(2)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TidyBot++ Wipe Table Task'
    )
    parser.add_argument(
        '--no-viewer', action='store_true',
        help='Run without MuJoCo viewer'
    )
    parser.add_argument(
        '--no-images', action='store_true',
        help='Disable image rendering'
    )
    parser.add_argument(
        '--show-images', action='store_true',
        help='Show camera images'
    )
    parser.add_argument(
        '--collect-data', action='store_true',
        help='Collect demonstration data'
    )
    parser.add_argument(
        '--output-dir', default='data/wipe_demos',
        help='Output directory for collected data'
    )
    parser.add_argument(
        '--num-episodes', type=int, default=1,
        help='Number of episodes to run'
    )
    main(parser.parse_args())
