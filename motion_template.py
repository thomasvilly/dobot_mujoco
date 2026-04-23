"""
Barebones motion script for quick experiments.

How to use this file:
1. Run it once as-is to watch a working pick-and-place sequence.
2. Change ONE thing in PHASES below.
3. Run it again and see what changed.

What to edit:
- `ctrl`: the 4 joint target values in radians.
- `suction_on`: whether the vacuum is on for that phase.
- `steps`: how long to hold that phase.

Easy first experiments:
- Change one joint value by +/- 0.05.
- Add 100 or 200 steps to a phase.
- Duplicate a phase and insert a pause.

Tip:
- Run `python teleop_keyboard.py`, move the arm where you want, then press Enter.
- The teleop script prints a `ctrl = np.array([...])` line you can paste here.
"""

import argparse
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from dobot_mujoco.env.dobot_pick_place import DobotPickPlace


# Each phase is:
# (name, ctrl_targets_for_motor1_to_motor4, suction_on, steps_to_hold)
PHASES = [
    ("home", np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), False, 300),
    (
        "pick_pump",
        np.array([-0.44271478, 0.67517413, 0.61999172, 1.30473021], dtype=np.float64),
        True,
        1500,
    ),
    (
        "move_to_place",
        np.array([0.61908900, 0.39283819, 0.33712828, 0.68738947], dtype=np.float64),
        True,
        1200,
    ),
    (
        "place",
        np.array([0.64473471, 0.59171374, 0.79252427, 0.61528668], dtype=np.float64),
        True,
        800,
    ),
    (
        "release",
        np.array([0.64473471, 0.59171374, 0.79252427, 0.61528668], dtype=np.float64),
        False,
        400,
    ),
]


def run_phase(env: DobotPickPlace, ctrl: np.ndarray, suction_on: bool, steps: int, viewer=None) -> None:
    env.suction_activated = suction_on
    for step in range(steps):
        env.data.ctrl[:4] = ctrl
        env.data.ctrl[4] = 1.0 if suction_on else 0.0
        mj.mj_step(env.model, env.data)
        if viewer is not None and step % 10 == 0:
            viewer.sync()
            time.sleep(0.002)


def main() -> None:
    parser = argparse.ArgumentParser(description="Editable motion-template script.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer.")
    args = parser.parse_args()

    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=args.seed)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for name, ctrl, suction_on, steps in PHASES:
            run_phase(env, ctrl, suction_on, steps, viewer=viewer)
            obs = env._get_obs()
            info = env._get_info(obs)
            cube_pos = env.data.body("pick_cube").xpos.copy()
            print(
                f"{name}: ctrl={ctrl} suction={suction_on} success={info['is_success']} "
                f"grasped={info['grasped']} cube_to_goal={info['cube_to_goal_distance']:.4f} "
                f"cube_pos={cube_pos}"
            )
            if viewer is not None:
                for _ in range(60):
                    viewer.sync()
                    time.sleep(0.005)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
