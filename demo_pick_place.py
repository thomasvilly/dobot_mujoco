import argparse
import time

import mujoco as mj
import numpy as np

from dobot_mujoco.env.dobot_pick_place import DobotPickPlace


HOME_CTRL = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
HOVER_PICK_CTRL = np.array([-0.43896897, 0.49773513, 0.08657091, 0.19996265], dtype=np.float64)
PICK_CTRL = np.array([-0.44271478, 0.67517413, 0.61999172, 1.30473021], dtype=np.float64)
HOVER_PLACE_CTRL = np.array([0.61908900, 0.39283819, 0.33712828, 0.68738947], dtype=np.float64)
PLACE_CTRL = np.array([0.64473471, 0.59171374, 0.79252427, 0.61528668], dtype=np.float64)


def run_phase(
    env: DobotPickPlace,
    target_ctrl: np.ndarray,
    suction_on: bool,
    steps: int,
    viewer=None,
    sync_every: int = 10,
) -> None:
    env.suction_activated = suction_on
    for step in range(steps):
        env.data.ctrl[:4] = target_ctrl
        env.data.ctrl[4] = 1.0 if suction_on else 0.0
        mj.mj_step(env.model, env.data)
        if viewer is not None and step % sync_every == 0:
            viewer.sync()
            time.sleep(0.002)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic Dobot pick-and-place demo.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=args.seed)

    phases = [
        ("home", HOME_CTRL, False, 300),
        ("pick_pump", PICK_CTRL, True, 1500),
        ("move_to_place", HOVER_PLACE_CTRL, True, 1200),
        ("place", PLACE_CTRL, True, 800),
        ("release", PLACE_CTRL, False, 400),
        ("retreat", HOVER_PLACE_CTRL, False, 400),
    ]

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for phase_name, target_ctrl, suction_on, steps in phases:
            run_phase(env, target_ctrl, suction_on, steps, viewer=viewer)
            obs = env._get_obs()
            info = env._get_info(obs)
            cube_pos = env.data.body("pick_cube").xpos.copy()
            print(
                f"{phase_name}: success={info['is_success']} grasped={info['grasped']} "
                f"cube_to_goal={info['cube_to_goal_distance']:.4f} cube_pos={cube_pos}"
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
