"""
MuJoCo motion template with a DOBOT-style student API.

This file mirrors the helper names used with the real robot as closely as we can:
- `initialize_robot(api)`
- `move_to_xyz(api, x, y, z)`
- `move_joint_angles(api, J1, J2, J3, J4=0)`
- `move_to_home(api)`
- `rotate_end_effector(api, angle)`
- `engage_suction(api)` / `release_suction(api)` / `stop_pump(api)`
- `get_pose(api)`

Important note:
- XYZ values here are in millimeters so the call shape matches the hardware code.
- The simulator still uses the simulator's world frame internally.

Easy first edits:
- Change one XYZ target below by +/- 10 mm.
- Add another `move_to_xyz(...)` between phases.
- Print `get_pose(api)` after a move to inspect the robot state.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from dobot_mujoco.env.dobot_cube_stack import DOBOT_MOTOR_LIMITS
from dobot_mujoco.env.dobot_pick_place import DobotPickPlace


EE_GEOM_NAME = "suctionCup_link2"
ARM_JOINT_NAMES = ["motor1", "motor2", "motor3", "motor4"]
ARM_LIMITS = np.array(DOBOT_MOTOR_LIMITS[:4], dtype=np.float64)
HOME_CTRL = np.zeros(4, dtype=np.float64)

# These joint targets were measured from the already-working scripted demo.
PICK_JOINTS = np.array([-24.3, 54.9, 39.9, 73.7], dtype=np.float64)
PLACE_LIFT_JOINTS = np.array([33.2, 30.9, 25.2, 40.1], dtype=np.float64)
PLACE_JOINTS = np.array([39.4, 43.2, 47.7, 38.9], dtype=np.float64)


@dataclass
class SimDobotAPI:
    env: DobotPickPlace
    viewer: object | None
    home_pos: np.ndarray
    suction_on: bool = False

    def sync_viewer(self, every: int = 1, step: int = 0) -> None:
        if self.viewer is not None and step % every == 0:
            self.viewer.sync()
            time.sleep(0.002)

    def current_xyz_m(self) -> np.ndarray:
        geom_id = self.env.model.geom(EE_GEOM_NAME).id
        return self.env.data.geom_xpos[geom_id].copy()

    def current_xyz_mm(self) -> np.ndarray:
        return self.current_xyz_m() * 1000.0

    def current_joint_deg(self) -> np.ndarray:
        joint_rad = np.array(
            [self.env.data.joint(name).qpos[0] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        return np.rad2deg(joint_rad)

    def get_pose(self) -> np.ndarray:
        xyz_mm = self.current_xyz_mm()
        joints_deg = self.current_joint_deg()
        tool_roll_deg = float(joints_deg[3])
        return np.array(
            [
                xyz_mm[0],
                xyz_mm[1],
                xyz_mm[2],
                tool_roll_deg,
                joints_deg[0],
                joints_deg[1],
                joints_deg[2],
                joints_deg[3],
            ],
            dtype=np.float64,
        )


def create_sim_api(seed: int, headless: bool) -> SimDobotAPI:
    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=seed)

    viewer = None
    if not headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    api = SimDobotAPI(env=env, viewer=viewer, home_pos=np.zeros(3, dtype=np.float64))
    api.home_pos = api.current_xyz_mm()
    return api


def get_pose(api: SimDobotAPI) -> np.ndarray:
    return api.get_pose()


def _step_env(api: SimDobotAPI, action: np.ndarray, max_steps: int, done_fn) -> None:
    for step in range(max_steps):
        api.env.step(action)
        api.sync_viewer(every=1, step=step)
        if done_fn():
            break


def _compute_cartesian_action(
    api: SimDobotAPI,
    desired_pos_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, data = api.env.model, api.env.data
    geom_id = model.geom(EE_GEOM_NAME).id
    dof_ids = [model.jnt_dofadr[model.joint(name).id] for name in ARM_JOINT_NAMES]

    jacp = np.zeros((3, model.nv))
    mj.mj_jacGeom(model, data, jacp, None, geom_id)
    jac = jacp[:, dof_ids]

    ee_pos = data.geom_xpos[geom_id].copy()
    pos_err = desired_pos_m - ee_pos
    cartesian_cmd = np.clip(8.0 * pos_err, -0.35, 0.35)
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), cartesian_cmd)

    action = np.zeros(5, dtype=np.float32)
    action[:4] = np.clip(dq / ARM_LIMITS, -1.0, 1.0)
    action[4] = 1.0 if api.suction_on else -1.0
    return action, ee_pos, pos_err


def initialize_robot(api: SimDobotAPI) -> None:
    api.suction_on = False
    api.env.suction_activated = False
    api.env.data.ctrl[:4] = HOME_CTRL
    api.env.data.ctrl[4] = 0.0
    for step in range(250):
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
    api.home_pos = api.current_xyz_mm()
    print(f"Simulator ready. home_pos = {api.home_pos.round(1).tolist()} mm")


def move_to_xyz(api: SimDobotAPI, x: float, y: float, z: float) -> None:
    # Best-effort Cartesian helper for the simulator. Joint-space motion is still
    # the most reliable path for the current MuJoCo model.
    target_m = np.array([x, y, z], dtype=np.float64) / 1000.0

    def done() -> bool:
        err_mm = np.linalg.norm(api.current_xyz_mm() - np.array([x, y, z], dtype=np.float64))
        return err_mm < 3.0

    for step in range(1200):
        action, _, _ = _compute_cartesian_action(api, target_m)
        api.env.step(action)
        api.sync_viewer(every=1, step=step)
        if done():
            break


def move_joint_angles(api: SimDobotAPI, J1: float, J2: float, J3: float, J4: float = 0.0) -> None:
    target_rad = np.deg2rad(np.array([J1, J2, J3, J4], dtype=np.float64))

    def done() -> bool:
        current_rad = np.array(
            [api.env.data.joint(name).qpos[0] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        err_deg = np.max(np.abs(np.rad2deg(current_rad - target_rad)))
        return err_deg < 1.0

    for step in range(1000):
        api.env.suction_activated = api.suction_on
        api.env.data.ctrl[:4] = target_rad
        api.env.data.ctrl[4] = 1.0 if api.suction_on else 0.0
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
        if done():
            break


def move_to_home(api: SimDobotAPI) -> None:
    move_joint_angles(api, 0.0, 0.0, 0.0, 0.0)


def rotate_end_effector(api: SimDobotAPI, angle: float) -> None:
    if -90.0 <= angle <= 90.0:
        j1, j2, j3, _ = api.current_joint_deg()
        move_joint_angles(api, j1, j2, j3, angle)


def engage_suction(api: SimDobotAPI) -> None:
    api.suction_on = True
    hold = np.zeros(5, dtype=np.float32)
    hold[4] = 1.0
    _step_env(api, hold, max_steps=60, done_fn=lambda: False)


def release_suction(api: SimDobotAPI) -> None:
    api.suction_on = False
    hold = np.zeros(5, dtype=np.float32)
    hold[4] = -1.0
    _step_env(api, hold, max_steps=60, done_fn=lambda: False)


def stop_pump(api: SimDobotAPI) -> None:
    release_suction(api)


def print_status(api: SimDobotAPI, label: str) -> None:
    obs = api.env._get_obs()
    info = api.env._get_info(obs)
    cube_pos = api.env.data.body("pick_cube").xpos.copy() * 1000.0
    pose = get_pose(api)
    print(
        f"{label}: xyz_mm={pose[:3].round(1)} joints_deg={pose[4:].round(1)} "
        f"suction={api.suction_on} grasped={info['grasped']} success={info['is_success']} "
        f"cube_to_goal={info['cube_to_goal_distance']:.4f} cube_mm={cube_pos.round(1)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT-style motion template for the MuJoCo simulator.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    api = create_sim_api(seed=args.seed, headless=args.headless)

    try:
        initialize_robot(api)
        print_status(api, "home")

        move_joint_angles(api, *PICK_JOINTS)
        print_status(api, "pick_pose")

        engage_suction(api)
        print_status(api, "pump_on")

        move_joint_angles(api, *PLACE_LIFT_JOINTS)
        print_status(api, "above_place")

        move_joint_angles(api, *PLACE_JOINTS)
        print_status(api, "at_place")

        release_suction(api)
        print_status(api, "released")

        move_joint_angles(api, *PLACE_LIFT_JOINTS)
        move_to_home(api)
        print_status(api, "back_home")
    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()


if __name__ == "__main__":
    main()
