from abc import abstractmethod
from typing import Optional

import mujoco
import gymnasium as gym
import numpy as np
import os

from gymnasium import spaces
from gymnasium_robotics.utils import mujoco_utils


DEFAULT_SIZE = 480


class BaseRobotEnv(gym.Env):
    """Superclass for all robotic environments."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path: str,
        initial_qpos,
        n_actions: int,
        decimation: int,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):
        """Initialize the hand and fetch robot superclass.

        Args:
            model_path (string): the path to the mjcf MuJoCo model.
            initial_qpos (np.ndarray): initial position value of the joints in the MuJoCo simulation.
            n_actions (integer): size of the action space.
            decimation (integer): number of MuJoCo simulation timesteps per Gymnasium step.
            render_mode (optional string): type of rendering mode, "human" for window rendeirng and "rgb_array" for offscreen. Defaults to None.
            width (optional integer): width of each rendered frame. Defaults to DEFAULT_SIZE.
            height (optional integer): height of each rendered frame . Defaults to DEFAULT_SIZE.
        """
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = os.path.join(
                os.path.dirname(__file__), "assets", model_path
            )
        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.n_substeps = decimation

        self.initial_qpos = initial_qpos

        self.width = width
        self.height = height
        self._initialize_simulation()

        self.goal = np.zeros(0)
        obs = self._get_obs()

        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self.render_mode = render_mode

    # Env methods
    # ----------------------------
    @abstractmethod
    def compute_reward(self, obs, info):
        """Compute the reward for the achieved goal with respect to the desired goal.

        Args:
            achieved_goal (np.ndarray): the goal that was achieved during execution.
            desired_goal (np.ndarray): the goal that we wanted to achieve.
            info (dictionary): an info dictionary containing auxiliary diagnostic information (helpful for debugging, learning, and logging).

        Returns:
            reward (float): The reward that corresponds to the provided achieved goal and desired goal.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_terminated(self, obs, info) -> bool:
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        raise NotImplementedError()

    @abstractmethod
    def compute_truncated(self, obs, info) -> bool:
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        raise NotImplementedError()

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._simulation_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = self._get_info(obs)

        terminated = self.compute_terminated(obs, info)
        truncated = self.compute_truncated(obs, info)

        reward = self.compute_reward(obs, info)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state. It should satisfy the `GoalEnv` :attr:`observation_space`.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        info = self._get_info(obs)
        if self.render_mode == "human":
            self.render()

        return obs, info

    # Extension methods
    # ----------------------------
    @abstractmethod
    def _simulation_step(self, action):
        """Advance the simulation.

        Override depending on the python bindings for the simulator
        """
        raise NotImplementedError()

    @abstractmethod
    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.

        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        return True

    @abstractmethod
    def _initialize_simulation(self):
        """Initialize simulation data structures."""
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Returns the observation."""
        raise NotImplementedError()

    def _get_info(self, observation) -> dict:
        """Returns the info dictionary."""
        return {}

    @abstractmethod
    def _set_action(self, action) -> None:
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    @abstractmethod
    def _is_success(self, obs, info) -> bool:
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    @abstractmethod
    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    @abstractmethod
    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment.

        Can be used to configure initial state and extract information from the simulation.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering.

        Can be used to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation.

        Can be used to enforce additional constraints on the simulation state.
        """
        pass


class MujocoRobotEnv(BaseRobotEnv):
    """Robot base class for fetch and hand environment versions that depend on new mujoco bindings from Deepmind."""

    def __init__(self, default_camera_config: Optional[dict] = None, **kwargs):
        """Initialize mujoco environment.

        The Deepmind mujoco bindings are initialized alongside the respective mujoco_utils.

        Args:
            default_camera_config (optional dictionary): dictionary of default mujoco camera parameters for human rendering. Defaults to None.
            The keys for this dictionary can be found in the mujoco mjvCamera struct:
            https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=azimuth#mjvcamera.

                - "type" (integer): camera type (mjtCamera)
                - "fixedcamid" (integer): fixed camera id
                - "trackbodyid": body id to track
                - "lookat" (np.ndarray): cartesian (x, y, z) lookat point
                - "distance" (float): distance to lookat point or tracked body
                - "azimuth" (float): camera azimuth (deg)
                - "elevation" (float): camera elevation (deg)

        Raises:
            error.DependencyNotInstalled: if mujoco bindings are not installed. Install with `pip install mujoco`
        """

        self._mujoco = mujoco
        self._utils = mujoco_utils
        self._default_camera_config = default_camera_config

        super().__init__(**kwargs)

        self.mujoco_renderer = None
        if self.render_mode is not None:
            self._ensure_renderer()

    def _ensure_renderer(self) -> None:
        if self.mujoco_renderer is not None:
            return

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            self._default_camera_config,
            width=self.width,
            height=self.height,
        )

    def _initialize_simulation(self):
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def _reset_sim(self):
        # Reset buffers for joint states, warm-start, control buffers etc.
        mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        return super()._reset_sim()

    def render(self):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._ensure_renderer()
        self._render_callback()
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnasium MujocoRenderer.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps

    def _simulation_step(self, action):
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
