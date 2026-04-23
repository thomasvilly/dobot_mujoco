import argparse
import time
from dataclasses import dataclass

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np

from dobot_mujoco.env.dobot_cube_stack import DOBOT_MOTOR_LIMITS
from dobot_mujoco.env.dobot_pick_place import DobotPickPlace


EE_GEOM_NAME = "suctionCup_link2"
ARM_JOINT_NAMES = ["motor1", "motor2", "motor3", "motor4"]
ARM_LIMITS = np.array(DOBOT_MOTOR_LIMITS[:4], dtype=np.float64)
WORKSPACE_MIN = np.array([-0.22, -0.24, 0.41], dtype=np.float64)
WORKSPACE_MAX = np.array([0.22, -0.05, 0.56], dtype=np.float64)


@dataclass
class TeleopState:
    desired_pos: np.ndarray
    home_pos: np.ndarray
    suction_on: bool = False
    reset_requested: bool = False
    print_requested: bool = False


def current_ee_position(env: DobotPickPlace) -> np.ndarray:
    geom_id = env.model.geom(EE_GEOM_NAME).id
    return env.data.geom_xpos[geom_id].copy()


def compute_cartesian_action(
    env: DobotPickPlace,
    desired_pos: np.ndarray,
    suction_on: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, data = env.model, env.data
    geom_id = model.geom(EE_GEOM_NAME).id
    dof_ids = [model.jnt_dofadr[model.joint(name).id] for name in ARM_JOINT_NAMES]

    jacp = np.zeros((3, model.nv))
    mj.mj_jacGeom(model, data, jacp, None, geom_id)
    jac = jacp[:, dof_ids]

    ee_pos = data.geom_xpos[geom_id].copy()
    pos_err = desired_pos - ee_pos
    cartesian_cmd = np.clip(8.0 * pos_err, -0.35, 0.35)
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), cartesian_cmd)

    action = np.zeros(5, dtype=np.float32)
    action[:4] = np.clip(dq / ARM_LIMITS, -1.0, 1.0)
    action[4] = 1.0 if suction_on else -1.0
    return action, ee_pos, pos_err


def hard_reset(env: DobotPickPlace) -> None:
    mj.mj_resetData(env.model, env.data)
    env.data.time = env.initial_time
    env.data.qpos[:] = np.copy(env.initial_qpos)
    env.data.qvel[:] = np.copy(env.initial_qvel)
    if env.model.na != 0:
        env.data.act[:] = None
    mj.mj_forward(env.model, env.data)
    env.previous_grasped = False
    env.grasped = False
    env.suction_activated = False


def print_status(env: DobotPickPlace, state: TeleopState) -> None:
    obs = env._get_obs()
    info = env._get_info(obs)
    cube = env.data.body("pick_cube").xpos.copy()
    goal = env.data.body("goal_marker").xpos.copy()
    ctrl = env.data.ctrl[:4].copy()
    ee = current_ee_position(env)
    print("Current status")
    print(f"  desired_xyz = {state.desired_pos}")
    print(f"  ee_xyz      = {ee}")
    print(f"  ctrl        = np.array({ctrl.tolist()}, dtype=np.float64)")
    print(f"  suction_on  = {state.suction_on}")
    print(f"  cube_xyz    = {cube}")
    print(f"  goal_xyz    = {goal}")
    print(
        f"  grasped={info['grasped']} success={info['is_success']} "
        f"cube_to_goal={info['cube_to_goal_distance']:.4f}"
    )


def make_key_callback(state: TeleopState, step_size: float):
    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_W:
            state.desired_pos[0] += step_size
        elif keycode == glfw.KEY_S:
            state.desired_pos[0] -= step_size
        elif keycode == glfw.KEY_A:
            state.desired_pos[1] -= step_size
        elif keycode == glfw.KEY_D:
            state.desired_pos[1] += step_size
        elif keycode == glfw.KEY_R:
            state.desired_pos[2] += step_size
        elif keycode == glfw.KEY_F:
            state.desired_pos[2] -= step_size
        elif keycode == glfw.KEY_SPACE:
            state.suction_on = not state.suction_on
            print(f"Suction {'ON' if state.suction_on else 'OFF'}")
        elif keycode == glfw.KEY_BACKSPACE:
            state.reset_requested = True
        elif keycode == glfw.KEY_H:
            state.desired_pos[:] = state.home_pos
            print(f"Target reset to home pose: {state.home_pos}")
        elif keycode == glfw.KEY_ENTER:
            state.print_requested = True

        state.desired_pos[:] = np.clip(state.desired_pos, WORKSPACE_MIN, WORKSPACE_MAX)

    return key_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for the Dobot scene.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.01,
        help="Target XYZ increment per key press, in metres.",
    )
    args = parser.parse_args()

    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=args.seed)

    initial_pos = current_ee_position(env)
    state = TeleopState(
        desired_pos=initial_pos.copy(),
        home_pos=initial_pos.copy(),
    )

    print("Keyboard teleop")
    print("  W/S: +X / -X")
    print("  A/D: -Y / +Y")
    print("  R/F: +Z / -Z")
    print("  Space: toggle suction")
    print("  H: return target to home pose")
    print("  Enter: print current actuator targets and pose")
    print("  Backspace: reset scene")
    print(f"  Starting EE pose: {initial_pos}")

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=make_key_callback(state, args.step_size),
    ) as viewer:
        while viewer.is_running():
            if state.reset_requested:
                hard_reset(env)
                state.reset_requested = False
                state.suction_on = False
                state.desired_pos[:] = current_ee_position(env)
                state.home_pos[:] = state.desired_pos
                print("Scene reset.")

            action, _, _ = compute_cartesian_action(env, state.desired_pos, state.suction_on)
            obs, reward, terminated, truncated, info = env.step(action)

            if state.print_requested:
                print_status(env, state)
                state.print_requested = False

            if terminated:
                print("Task success reached. Press Backspace to reset or keep driving.")

            viewer.sync()
            time.sleep(0.002)

    env.close()


if __name__ == "__main__":
    main()
