import mujoco as mj
import numpy as np

from .dobot_cube_stack import (
    DobotCubeStack,
    EE_LINK_NAME,
    bodies_are_colliding,
)


CUBE_HALF_EXTENT = 0.02
TABLE_TOP_Z = 0.39
PICK_POS = np.array([-0.121, -0.1567, TABLE_TOP_Z], dtype=np.float64)
GOAL_POS = np.array([0.161, -0.11, TABLE_TOP_Z], dtype=np.float64)
SUCCESS_TOLERANCE = 0.035
MIN_LIFT_HEIGHT = TABLE_TOP_Z + 0.015
TARGET_LIFT_HEIGHT = TABLE_TOP_Z + 0.025


class DobotPickPlace(DobotCubeStack):
    def __init__(
        self,
        default_camera_config=None,
        position_jitter: float = 0.0,
        goal_distance_scale: float = 1.0,
        success_tolerance: float = SUCCESS_TOLERANCE,
        **kwargs,
    ) -> None:
        self.position_jitter = position_jitter
        self.goal_distance_scale = float(np.clip(goal_distance_scale, 0.2, 1.0))
        self.success_tolerance = float(success_tolerance)
        self._prev_reward_metrics = None
        super().__init__(default_camera_config=default_camera_config, n_cubes=1, **kwargs)

    def _is_success(self, obs, info) -> bool:
        return (
            info["cube_to_goal_distance"] < self.success_tolerance
            and not info["grasped"]
            and not info["suction_activated"]
        )

    def compute_reward(self, obs, info):
        reach_shaping = 1.0 - np.tanh(8.0 * info["ee_to_cube_distance"])
        goal_shaping = 1.0 - np.tanh(5.0 * info["cube_to_goal_distance"])
        reward = -0.02

        if not info["grasped"]:
            reward += 1.5 * reach_shaping
            if self._prev_reward_metrics is not None:
                reward += 20.0 * (
                    self._prev_reward_metrics["ee_to_cube_distance"]
                    - info["ee_to_cube_distance"]
                )

        if self._prev_reward_metrics is not None and info["grasped"]:
            reward += 100.0 * (
                info["cube_height"] - self._prev_reward_metrics["cube_height"]
            )

        if info["grasped"] and not self.previous_grasped:
            reward += 10.0

        if info["grasped"] and not info["is_lifted"]:
            reward -= 0.25

        reward += 5.0 * info["lift_fraction"]
        if info["is_lifted"]:
            reward += 2.0
            reward += 3.0 * goal_shaping
            if self._prev_reward_metrics is not None:
                reward += 30.0 * (
                    self._prev_reward_metrics["cube_to_goal_distance"]
                    - info["cube_to_goal_distance"]
                )

        if self.previous_grasped and not info["grasped"] and not info["is_success"]:
            reward -= 8.0

        if self._is_success(obs, info):
            reward += 40.0

        self._prev_reward_metrics = {
            "ee_to_cube_distance": info["ee_to_cube_distance"],
            "cube_to_goal_distance": info["cube_to_goal_distance"],
            "cube_height": info["cube_height"],
            "is_lifted": info["is_lifted"],
        }
        return reward

    def compute_terminated(self, obs, info) -> bool:
        return self._is_success(obs, info)

    def _reset_sim(self) -> bool:
        is_valid = super()._reset_sim()
        self._prev_reward_metrics = None
        return is_valid

    def _randomize_spec(self):
        self.randomized_spec = self.mjspec.copy()

        cube_pos = PICK_POS.copy()
        goal_delta = GOAL_POS - PICK_POS
        goal_pos = PICK_POS + self.goal_distance_scale * goal_delta

        if self.position_jitter > 0.0:
            jitter = self.np_random.uniform(
                low=-self.position_jitter,
                high=self.position_jitter,
                size=(2, 2),
            )
            cube_pos[:2] += jitter[0]
            goal_pos[:2] += jitter[1]

        cube_body = self.randomized_spec.worldbody.add_body(name="pick_cube")
        cube_body.pos = cube_pos
        cube_body.add_freejoint(name="pick_cube_freejoint")
        cube_geom = cube_body.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF_EXTENT, CUBE_HALF_EXTENT, CUBE_HALF_EXTENT],
            rgba=[0.2, 0.45, 0.95, 1.0],
            mass=0.04,
        )
        cube_geom.solimp = np.array([1, 1, 0.001, 0.5, 2])
        cube_geom.solref = np.array([0.002, 1.0])

        goal_body = self.randomized_spec.worldbody.add_body(name="goal_marker")
        goal_body.pos = goal_pos
        goal_body.add_geom(
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.03, 0.03, 0.002],
            rgba=[0.15, 0.85, 0.2, 0.35],
            contype=0,
            conaffinity=0,
        )
        goal_body.add_site(
            name="goal_site",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.01, 0.01],
            rgba=[0.15, 0.85, 0.2, 0.9],
        )

        self.model = self.randomized_spec.compile()
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        return not bodies_are_colliding(self.model, self.data, EE_LINK_NAME, "pick_cube")

    def _sample_goal(self):
        return self.data.body("goal_marker").xpos.copy()

    def _get_obs(self):
        suction_activated = float(getattr(self, "suction_activated", False))
        grasped = float(getattr(self, "grasped", False))
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
        ee_pos = self.data.body(EE_LINK_NAME).xpos.copy()
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

        self.ee_pos = ee_pos
        self.cube_pos = self.data.body("pick_cube").xpos.copy()
        self.goal_pos = self.data.body("goal_marker").xpos.copy()

        return np.concatenate(
            (
                qpos,
                qdot,
                ee_pos,
                ee_vel,
                self.cube_pos,
                self.goal_pos,
                self.cube_pos - ee_pos,
                self.goal_pos - self.cube_pos,
                np.array([suction_activated, grasped]),
            )
        ).astype(np.float32)

    def _get_info(self, observation):
        cube_top = self.cube_pos + np.array([0.0, 0.0, CUBE_HALF_EXTENT])
        cube_to_goal_distance = np.linalg.norm(self.cube_pos - self.goal_pos)
        ee_to_cube_distance = np.linalg.norm(self.ee_pos - cube_top)
        cube_height_above_table = max(0.0, self.cube_pos[2] - TABLE_TOP_Z)
        is_ee_on_cube = bodies_are_colliding(self.model, self.data, EE_LINK_NAME, "pick_cube")

        self.previous_grasped = self.grasped
        self.grasped = is_ee_on_cube and self.suction_activated

        lift_fraction = np.clip(
            cube_height_above_table / (TARGET_LIFT_HEIGHT - TABLE_TOP_Z),
            0.0,
            1.0,
        )
        is_lifted = self.cube_pos[2] >= MIN_LIFT_HEIGHT

        info = {
            "grasped": self.grasped,
            "suction_activated": self.suction_activated,
            "cube_to_goal_distance": cube_to_goal_distance,
            "ee_to_cube_distance": ee_to_cube_distance,
            "cube_height": self.cube_pos[2],
            "cube_height_above_table": cube_height_above_table,
            "lift_fraction": lift_fraction,
            "is_lifted": is_lifted,
            "goal_distance_scale": self.goal_distance_scale,
        }
        info.update({"is_success": self._is_success(observation, info)})
        return info
