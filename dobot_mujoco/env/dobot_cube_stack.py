import os

import numpy as np
import mujoco as mj
from .base_env import MujocoRobotEnv
from gymnasium import spaces


def color_range(s: float, v: float, n: int):
    """
    Creates an `n` size discrete color range based on Value and Saturation passed.

    params:
        - s: float  Value in [0, 1]
        - v: float  Saturation in [0, 1]
    """

    s = np.clip(s, 0, 1)
    v = np.clip(v, 0, 1)

    colors = []

    hues = np.linspace(0.0, 1.0, n + 1)

    for hue in hues[:-1]:

        index = int(hue * 6)
        f = hue * 6 - index
        q = 1 - f

        if index % 6 == 0 or index == 0:
            colors.append([1, f, 0])
        if index == 1:
            colors.append([q, 1, 0])
        if index == 2:
            colors.append([0, 1, f])
        if index == 3:
            colors.append([0, q, 1])
        if index == 4:
            colors.append([f, 0, 1])
        if index == 5:
            colors.append([1, 0, q])

    return colors


R_MIN = 0.15  # meters
R_MAX = 0.275  # meters
THETA_LIM = np.pi / 3  # rad
TABLE_THICKNESS = 0.02  # meters

PI2_ROT = np.array(
    [
        [np.cos(-np.pi / 2), -np.sin(np.pi / 2)],
        [np.sin(-np.pi / 2), np.cos(np.pi / 2)],
    ]
)

DOBOT_MOTOR_LIMITS = [
    5.59,  # rad/s
    5.59,  # rad/s
    5.59,  # rad/s
    8.38,  # rad/s
    3.0,  # Has no meaning, the rate at which the pump activates ?
]

DOBOT_ACT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_lift",
    "tool_roll",
    "suction_cup_pump",
]

EE_LINK_NAME = "suctionCup_link2"


def resolve_body_name(model: mj.MjModel, body_name: str) -> str:
    """Handle attached-body names, which may be prefixed with '/' at runtime."""
    try:
        model.body(body_name)
        return body_name
    except KeyError:
        alt_name = f"/{body_name}"
        model.body(alt_name)
        return alt_name


def bodies_are_colliding(
    model: mj.MjModel, data: mj.MjData, body1_name: str, body2_name: str
) -> bool:
    """Check if two bodies are colliding."""
    body1_name = resolve_body_name(model, body1_name)
    body2_name = resolve_body_name(model, body2_name)
    body1_id = model.body(body1_name).id
    body2_id = model.body(body2_name).id

    # Get all geoms that belong to each body
    geom1_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body1_id]
    geom2_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body2_id]

    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in geom1_ids and c.geom2 in geom2_ids) or (
            c.geom1 in geom2_ids and c.geom2 in geom1_ids
        ):
            return True
    return False


class DobotCubeStack(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 40,
    }

    def __init__(
        self,
        default_camera_config=None,
        n_cubes=2,
        **kwargs,
    ) -> None:
        self.n_cubes = n_cubes
        self.mujoco_renderer = None
        super().__init__(
            default_camera_config=default_camera_config,
            model_path=os.path.join(
                os.path.dirname(__file__), "assets", "dobot_table_scene.xml"
            ),
            decimation=25,  # 25 mujoco steps per env step
            initial_qpos=0,
            n_actions=5,
            **kwargs,
        )
        if self.render_mode is not None:
            self.render()

    def _set_action(self, action) -> None:
        q_target = np.copy(self.data.ctrl)

        delta_q = action * DOBOT_MOTOR_LIMITS * self.dt
        q_target += delta_q

        # update joint targets
        q_target = np.clip(q_target, *(self.jnt_ranges.T))
        q_target[-1] = (
            action[-1] > 0.2 if not self.suction_activated else action[-1] > -0.2
        )  # suction cup control
        self.suction_activated = bool(q_target[-1])

        self.data.ctrl = q_target

    def _is_success(self, obs, info) -> bool:

        return (
            info["is_above_target"]
            and info["is_cube_a_on_cube_b"]
            and not info["grasped"]
        )

    def compute_reward(self, obs, info):
        grasped = info.get("grasped", False)
        d1 = info["ee_to_cube_a_distance"]
        d2 = info["distance_to_target"]

        reward = -d1 - d2
        if grasped and not self.previous_grasped:
            # Reward for successfully grasping the cube
            reward += 2.0
        elif (
            not grasped and self.previous_grasped and info["distance_to_target"] > 0.02
        ):
            # Penalize dropping the cube if the cube is far from the target
            reward -= 2.0

        if info["is_success"]:
            reward += 10.0

        return reward

    def compute_terminated(self, obs, info) -> bool:
        return self._is_success(obs, info)

    def compute_truncated(self, obs, info) -> bool:
        return False

    def _initialize_simulation(self):
        self.mjspec: mj.MjSpec = mj.MjSpec.from_file(self.fullpath)
        self.jnt_ranges = np.array(
            [self.mjspec.actuator(name).ctrlrange for name in DOBOT_ACT_NAMES]
        )
        self.act_targets = [
            self.mjspec.actuator(name).target for name in DOBOT_ACT_NAMES
        ]
        self.jnt_names = [jnt.name for jnt in self.mjspec.joints]
        self.grasped = False

        self._randomize_spec()

    def _randomize_spec(self):
        self.randomized_spec = self.mjspec.copy()

        cube_specs = self._randomize_cube_domain()
        cubes_frame = self.randomized_spec.worldbody.add_frame(
            name="cubes_frame", pos=[0, 0, 0.177 * 2]
        )

        for i, cube_spec in enumerate(cube_specs):
            self.randomized_spec.attach(cube_spec, suffix=f"_{i}", frame=cubes_frame)

        self.model: mj.MjModel = self.randomized_spec.compile()
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        return not bodies_are_colliding(self.model, self.data, "cube_0", "cube_1")

    def _reset_sim(self) -> bool:
        is_valid = self._randomize_spec()
        self.previous_grasped = False
        self.grasped = False
        self.suction_activated = False

        # Weird hack to update the viewer when randomizing the domain
        # and recompiling the randomized spec
        if self.mujoco_renderer is not None:
            viewer = self.mujoco_renderer._viewers.get(self.render_mode)
            if viewer is not None:
                viewer.model = self.model
                viewer.data = self.data

        return is_valid

    def _sample_goal(self):
        desired_goal_grasp = self.data.body(resolve_body_name(self.model, "cube_0")).xpos + np.array([0, 0, 0.02])
        desired_goal_cube = self.data.body(resolve_body_name(self.model, "cube_1")).xpos + np.array([0, 0, 0.02])
        return np.concatenate((desired_goal_grasp, desired_goal_cube))

    def _get_obs(self):
        qpos = [
            self.data.joint(name).qpos[0]
            for name in self.act_targets
            if name in self.jnt_names
        ]
        qdot = [
            self.data.joint(name).qvel[0]
            for name in self.act_targets
            if name in self.jnt_names
        ]
        ee_pos = self.data.body(EE_LINK_NAME).xpos
        ee_vel = np.zeros(6)
        mj.mj_objectVelocity(
            self.model,
            self.data,
            mj.mjtObj.mjOBJ_BODY,
            self.model.body(EE_LINK_NAME).id,
            ee_vel,
            0,
        )
        ee_vel = ee_vel[3:]
        self.ee_pos = ee_pos.copy()
        cube_a_name = resolve_body_name(self.model, "cube_0")
        cube_b_name = resolve_body_name(self.model, "cube_1")
        self.cube_a_pos = self.data.body(cube_a_name).xpos.copy()
        self.cube_a_quat = self.data.body(cube_a_name).xquat.copy()
        self.cube_b_pos = self.data.body(cube_b_name).xpos.copy()
        self.cube_b_quat = self.data.body(cube_b_name).xquat.copy()

        return np.concatenate(
            (
                qpos,
                qdot,
                ee_pos,
                ee_vel,
                self.cube_a_pos,
                self.cube_a_quat,
                self.cube_b_pos,
                self.cube_b_quat,
            )
        ).astype(np.float32)

    def _get_info(self, observation):
        info = super()._get_info(observation)

        target_pos = self.cube_b_pos + np.array([0, 0, 0.02])
        cube_a_bottom = self.cube_a_pos - np.array([0, 0, 0.02])
        is_above_target = cube_a_bottom[2] >= target_pos[2]
        is_cube_a_on_cube_b = bodies_are_colliding(
            self.model, self.data, "cube_0", "cube_1"
        )
        is_ee_on_cube_a = bodies_are_colliding(
            self.model, self.data, EE_LINK_NAME, "cube_0"
        )
        self.previous_grasped = self.grasped
        self.grasped = is_ee_on_cube_a and self.suction_activated

        info.update(
            {
                "grasped": self.grasped,
                "suction_activated": self.suction_activated,
                "is_above_target": is_above_target,
                "is_cube_a_on_cube_b": is_cube_a_on_cube_b,
                "distance_to_target": np.linalg.norm(cube_a_bottom - target_pos),
                "ee_to_cube_a_distance": np.linalg.norm(
                    (self.cube_a_pos + np.array([0, 0, 0.02])) - self.ee_pos
                ),
            }
        )
        info.update({"is_success": self._is_success(observation, info)})
        return info

    def _randomize_cube_domain(self) -> list[mj.MjSpec]:
        """
        Creates a list of `MjSpec` with randomized cubes and returns it
        """
        specs = []
        colors = color_range(0.8, 1.0, self.n_cubes)
        for i in range(self.n_cubes):
            cube_spec = mj.MjSpec()
            cube_spec.compiler.degree = False

            # Sample cube position relative to Dobot in polar coordinates
            r = np.random.uniform(R_MIN, R_MAX)
            theta = np.random.uniform(-THETA_LIM, THETA_LIM)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coords = np.array([x, y])
            coords_rot = PI2_ROT @ coords.T

            cube_body = cube_spec.worldbody.add_body(name=f"cube")

            cube_body.pos = np.array([*coords_rot, TABLE_THICKNESS + 0.01])
            geom = cube_body.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.02, 0.02],  # m
                rgba=[*colors[i], 1.0],
                mass=np.random.uniform(0.01, 0.1),  # kg
            )
            cube_body.add_freejoint()
            geom.solimp = np.array([1, 1, 0.001, 0.5, 2])
            geom.solref = np.array([0.002, 1.0])

            specs.append(cube_spec)

        return specs


if __name__ == "__main__":
    env = DobotCubeStack(render_mode="human")

    env.reset(seed=1234)

    while True:
        act = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, r, terminated, truncated, info = env.step(act)
        print(f"Observation : {obs}")
        print(f"Reward : {r}")
        print(f"Terminated : {terminated}")
        print(f"truncated : {truncated}")
        print(f"Info : {info}")

        if terminated or truncated:
            env.reset()
