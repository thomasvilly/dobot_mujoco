import argparse
import os
import select
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass

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


@dataclass
class SimulationSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray | None
    time: float


class TerminalKeyReader(AbstractContextManager["TerminalKeyReader"]):
    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._fd: int | None = None
        self._old_settings = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> "TerminalKeyReader":
        if not self._enabled:
            return self

        if os.name == "nt":
            return self

        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._enabled or os.name == "nt" or self._fd is None or self._old_settings is None:
            return None

        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        return None

    def read_key(self) -> str | None:
        if not self._enabled:
            return None

        if os.name == "nt":
            import msvcrt

            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                # Swallow the second code unit for function / arrow keys.
                if msvcrt.kbhit():
                    msvcrt.getwch()
                return None
            return key

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None

        key = sys.stdin.read(1)
        if key == "\x1b":
            # Drain the rest of an escape sequence (arrows, etc.) and ignore it.
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if not ready:
                    break
                sys.stdin.read(1)
            return None

        return key


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


def capture_sim_snapshot(env: DobotPickPlace) -> SimulationSnapshot:
    act = env.data.act.copy() if env.model.na != 0 else None
    return SimulationSnapshot(
        qpos=env.data.qpos.copy(),
        qvel=env.data.qvel.copy(),
        ctrl=env.data.ctrl.copy(),
        act=act,
        time=float(env.data.time),
    )


def hard_reset(env: DobotPickPlace, snapshot: SimulationSnapshot) -> None:
    mj.mj_resetData(env.model, env.data)
    env.data.time = snapshot.time
    env.data.qpos[:] = snapshot.qpos
    env.data.qvel[:] = snapshot.qvel
    env.data.ctrl[:] = snapshot.ctrl
    if env.model.na != 0 and snapshot.act is not None:
        env.data.act[:] = snapshot.act
    mj.mj_forward(env.model, env.data)
    env.goal = env.data.body("goal_marker").xpos.copy()
    env.previous_grasped = False
    env.grasped = False
    env.suction_activated = False


def handle_terminal_key(state: TeleopState, key: str, step_size: float) -> bool:
    key_lower = key.lower()

    if key_lower == "w":
        state.desired_pos[0] += step_size
    elif key_lower == "s":
        state.desired_pos[0] -= step_size
    elif key_lower == "a":
        state.desired_pos[1] -= step_size
    elif key_lower == "d":
        state.desired_pos[1] += step_size
    elif key_lower == "r":
        state.desired_pos[2] += step_size
    elif key_lower == "f":
        state.desired_pos[2] -= step_size
    elif key == " ":
        state.suction_on = not state.suction_on
        print(f"Suction {'ON' if state.suction_on else 'OFF'}")
    elif key_lower == "h":
        state.desired_pos[:] = state.home_pos
        print(f"Target reset to home pose: {state.home_pos}")
    elif key in ("\r", "\n"):
        state.print_requested = True
    elif key in ("\x08", "\x7f"):
        state.reset_requested = True
    elif key_lower == "q":
        return True

    state.desired_pos[:] = np.clip(state.desired_pos, WORKSPACE_MIN, WORKSPACE_MAX)
    return False


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
    initial_snapshot = capture_sim_snapshot(env)

    initial_pos = current_ee_position(env)
    state = TeleopState(
        desired_pos=initial_pos.copy(),
        home_pos=initial_pos.copy(),
    )

    print("Keyboard teleop (terminal-driven)")
    print("  Keep the terminal focused while the MuJoCo window is open.")
    print("  W/S: +X / -X")
    print("  A/D: -Y / +Y")
    print("  R/F: +Z / -Z")
    print("  Space: toggle suction")
    print("  H: return target to home pose")
    print("  Enter: print current actuator targets and pose")
    print("  Backspace: reset scene")
    print("  Q: quit teleop")
    print(f"  Starting EE pose: {initial_pos}")

    with TerminalKeyReader() as key_reader:
        if not key_reader.enabled:
            raise RuntimeError("teleop_keyboard.py needs to be run from an interactive terminal.")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            should_quit = False
            while viewer.is_running() and not should_quit:
                while True:
                    key = key_reader.read_key()
                    if key is None:
                        break
                    should_quit = handle_terminal_key(state, key, args.step_size)
                    if should_quit:
                        break

                if state.reset_requested:
                    hard_reset(env, initial_snapshot)
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
