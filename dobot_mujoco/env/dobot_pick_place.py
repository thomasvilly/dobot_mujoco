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
LIFT_HEIGHT = TABLE_TOP_Z + 0.04


class DobotPickPlace(DobotCubeStack):
    def __init__(self, default_camera_config=None, position_jitter: float = 0.0, **kwargs) -> None:
        self.position_jitter = position_jitter
        super().__init__(default_camera_config=default_camera_config, n_cubes=1, **kwargs)

    def _is_success(self, obs, info) -> bool:
        return (
            info["cube_to_goal_distance"] < SUCCESS_TOLERANCE
            and not info["grasped"]
            and not info["suction_activated"]
        )

    def compute_reward(self, obs, info):
        reward = -0.6 * info["ee_to_cube_distance"] - 1.2 * info["cube_to_goal_distance"]
        reward += 2.0 * info["lift_fraction"]

        if info["grasped"] and not self.previous_grasped:
            reward += 4.0

        if info["cube_height"] > LIFT_HEIGHT:
            reward += 1.5

        if self._is_success(obs, info):
            reward += 12.0

        return reward

    def compute_terminated(self, obs, info) -> bool:
        return self._is_success(obs, info)

    def _randomize_spec(self):
        self.randomized_spec = self.mjspec.copy()

        cube_pos = PICK_POS.copy()
        goal_pos = GOAL_POS.copy()

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
            size=[0.03, 0.002],
            rgba=[0.15, 0.85, 0.2, 0.35],
            contype=0,
            conaffinity=0,
        )
        goal_body.add_site(
            name="goal_site",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.01],
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
        is_ee_on_cube = bodies_are_colliding(self.model, self.data, EE_LINK_NAME, "pick_cube")

        self.previous_grasped = self.grasped
        self.grasped = is_ee_on_cube and self.suction_activated

        lift_fraction = np.clip((self.cube_pos[2] - TABLE_TOP_Z) / 0.08, 0.0, 1.0)

        info = {
            "grasped": self.grasped,
            "suction_activated": self.suction_activated,
            "cube_to_goal_distance": cube_to_goal_distance,
            "ee_to_cube_distance": ee_to_cube_distance,
            "cube_height": self.cube_pos[2],
            "lift_fraction": lift_fraction,
        }
        info.update({"is_success": self._is_success(observation, info)})
        return info
