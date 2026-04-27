# Dobot MuJoCo Codebase Guide

This note is a handoff-style map of the repository with a bias toward two questions:

1. Where is the simulation logic?
2. If we want to give the lab a smaller "simulation-only" version, what can we remove safely?

## Short Answer

- The actual MuJoCo robot/world model lives in:
  - `dobot_mujoco/env/assets/dobot_table_scene.xml`
  - `dobot_mujoco/env/assets/dobot_suction_cup.xml`
  - `dobot_mujoco/env/assets/meshes/`
- The Gym-compatible simulator wrapper lives in:
  - `dobot_mujoco/env/base_env.py`
  - `dobot_mujoco/env/dobot_cube_stack.py`
  - `dobot_mujoco/env/dobot_pick_place.py`
- The simplest viewer-only entrypoint is:
  - `main.py`
- The simplest scripted manipulation entrypoints are:
  - `demo_pick_place.py`
  - `motion_template.py`
- The RL-specific files are:
  - `train.py`
  - `record_rollouts.py`
  - `dobot_mujoco/rollout_recording.py`

## Where The IK Is

There is **no central IK module** in this repo.

The only real IK-like logic is a small Cartesian helper in `motion_template.py:119-139`.

It does:

- MuJoCo geom Jacobian query with `mj.mj_jacGeom(...)`
- a damped least-squares solve
- conversion from Cartesian error to a 5D action vector

That helper is:

- `_compute_cartesian_action(...)` in `motion_template.py:119-139`
- used by `move_to_xyz(...)` in `motion_template.py:154-168`

Important context:

- This is **not** used by the RL envs.
- This is **not** used by `main.py`.
- This is **not** the primary control path for the current simulator.
- Even the file itself says joint-space motion is still more reliable:
  - `motion_template.py:155-156`

So if you are asking "where does the simulator do IK by default?", the answer is:

- It mostly does **not**.
- The default env/action interface is joint-space control.
- The one Cartesian/IK helper is just a convenience layer in `motion_template.py`.

## High-Level Layout

### Core simulation package

- `dobot_mujoco/__init__.py`
  - imports `dobot_mujoco.env` so the Gym envs register on import
- `dobot_mujoco/env/__init__.py:3-10`
  - registers:
    - `DobotCubeStack-v0`
    - `DobotPickPlace-v0`

### Environment base classes

- `dobot_mujoco/env/base_env.py`
  - generic Gymnasium-style MuJoCo env base
  - owns `step`, `reset`, rendering, and simulation stepping
  - `BaseRobotEnv.step(...)`: `base_env.py:108-144`
  - `BaseRobotEnv.reset(...)`: `base_env.py:146-179`
  - `MujocoRobotEnv._simulation_step(...)`: `base_env.py:351-352`

### Task/env implementations

- `dobot_mujoco/env/dobot_cube_stack.py`
  - "base Dobot env" for the repo
  - defines:
    - actuator names
    - motor limits
    - normalized action-to-joint-target mapping
    - collision-based grasp logic
    - random cube insertion via `MjSpec`
- `dobot_mujoco/env/dobot_pick_place.py`
  - derived single-cube pick/place task
  - defines:
    - pick/goal positions
    - reward shaping
    - success condition
    - simplified observation/info dictionary

### MuJoCo assets

- `dobot_mujoco/env/assets/dobot_table_scene.xml`
  - table, ground, light, camera, and robot attachment point
- `dobot_mujoco/env/assets/dobot_suction_cup.xml`
  - the Dobot arm, joints, tool, actuator definitions
- `dobot_mujoco/env/assets/meshes/`
  - STL meshes referenced by `dobot_suction_cup.xml`

### Scripts / entrypoints

- `main.py`
  - raw viewer smoke test
  - no Gym env wrapper
- `demo_pick_place.py`
  - hard-coded joint-target pick/place sequence
- `motion_template.py`
  - "student API" wrapper for scripted experiments
  - contains the only IK-like Cartesian helper
- `train.py`
  - PPO/CrossQ training driver
- `record_rollouts.py`
  - replay/export tool for trained checkpoints

### Real robot utilities

- `dobot_firm_utils/`
  - hardware-side helper scripts
  - not imported by the simulator code
  - safe to remove for a simulation-only handoff

### Generated / experiment output

- `models/`
- `logs/`
- `recordings/`

These are outputs, not source.

## What Actually Defines The Robot

There are two layers:

### 1. World / scene file

`dobot_mujoco/env/assets/dobot_table_scene.xml`

This file defines:

- MuJoCo timestep and solver options: `dobot_table_scene.xml:1-4`
- the table and ground: `dobot_table_scene.xml:24-58`
- where the robot is mounted on the table: `dobot_table_scene.xml:48-51`
- a camera named `front`: `dobot_table_scene.xml:56-57`

It imports the robot model as an asset:

- `dobot_table_scene.xml:20-21`

### 2. Robot / actuator file

`dobot_mujoco/env/assets/dobot_suction_cup.xml`

This file defines the Dobot arm and suction tool itself.

Key things to know:

- `motor3` joint: `dobot_suction_cup.xml:350-351`
- `motor4` joint: `dobot_suction_cup.xml:307-308`
- end-effector body `suctionCup_link2`: `dobot_suction_cup.xml:315-342`
- actuator block: `dobot_suction_cup.xml:394-405`

Actuators are:

- `shoulder_pan`
- `shoulder_lift`
- `elbow_lift`
- `tool_roll`
- `suction_cup_pump`

Those actuator names are mirrored in Python in:

- `dobot_mujoco/env/dobot_cube_stack.py:67-73`

The suction is modeled as an MuJoCo adhesion actuator:

- `dobot_suction_cup.xml:404`

## How Control Works

There are three control styles in this repo.

### 1. Raw MuJoCo stepping

Used in:

- `main.py:5-14`

This just loads the XML model and repeatedly calls `mujoco.mj_step`.

Use this when:

- you only want to verify the model/viewer
- you do not care about Gym, rewards, or tasks

### 2. Env action space control

Used in:

- RL
- any code that calls `env.step(action)`

The env action space is a 5D normalized vector in `[-1, 1]`:

- 4 joint controls
- 1 suction control

The conversion happens in `dobot_cube_stack.py:141-154`:

- start from current control target `self.data.ctrl`
- compute `delta_q = action * DOBOT_MOTOR_LIMITS * self.dt`
- add delta to current actuator targets
- clip to actuator ranges
- interpret the last action dimension as suction on/off hysteresis

This is the main control path used by the learning code.

### 3. Direct joint target scripting

Used in:

- `demo_pick_place.py:17-32`
- `motion_template.py:171-189`

These scripts bypass the normalized action interface and write directly to:

- `env.data.ctrl[:4]`
- `env.data.ctrl[4]`

This is often the simplest way to script deterministic motions in this repo.

## Where The Pick/Place Task Is Defined

`dobot_mujoco/env/dobot_pick_place.py`

This file defines the single-cube task used by most recent work.

### Task geometry

- cube start pose constant: `dobot_pick_place.py:13`
- goal pose constant: `dobot_pick_place.py:14`
- success threshold: `dobot_pick_place.py:15`
- lift thresholds: `dobot_pick_place.py:16-17`

### Success condition

`dobot_pick_place.py:35-40`

Success means:

- cube is close to goal
- robot is not grasping it
- suction is off

### Reward

`dobot_pick_place.py:42-88`

This is where almost all RL behavior shaping lives:

- reach shaping
- lift shaping
- grasp bonus
- failure penalties
- success bonus

If training is weird, this is one of the first files to inspect.

### Observation contents

`dobot_pick_place.py:151-192`

Observation includes:

- joint positions
- joint velocities
- end-effector position and velocity
- cube position
- goal position
- cube minus EE vector
- goal minus cube vector
- suction/grasp flags

### Info dictionary

`dobot_pick_place.py:194-223`

This is where debugging metrics come from:

- `grasped`
- `suction_activated`
- `cube_to_goal_distance`
- `ee_to_cube_distance`
- `cube_height`
- `cube_height_above_table`
- `lift_fraction`
- `is_lifted`

## How The Base Env Is Built

The base control/render/reset flow is in `dobot_mujoco/env/base_env.py`.

Important pieces:

- `BaseRobotEnv.step(...)`: `base_env.py:108-144`
- `BaseRobotEnv.reset(...)`: `base_env.py:146-179`
- lazy renderer creation: `base_env.py:288-300`
- render method: `base_env.py:328-336`
- MuJoCo multi-step advance: `base_env.py:351-352`

One subtle but important detail:

- `MujocoRobotEnv._initialize_simulation(...)` in `base_env.py:302-313`
  loads a plain XML model with `MjModel.from_xml_path(...)`
- but `DobotCubeStack` **overrides** `_initialize_simulation(...)`
  with its own `MjSpec`-based pipeline in `dobot_cube_stack.py:190-218`

So for the real task envs, the simulation is not just "load XML once and go".
The env recompiles a model from `MjSpec` so it can inject cubes and goals programmatically.

## Where Randomization Happens

There are two styles of scene construction:

### Static scene

`main.py` uses the static XML directly:

- `main.py:6-9`

### Programmatically compiled task scene

The envs use `MjSpec` so they can insert task objects.

In `dobot_cube_stack.py`:

- base spec from file: `dobot_cube_stack.py:190-198`
- randomized copy: `dobot_cube_stack.py:203-218`
- random cube creation: `dobot_cube_stack.py:314-348`

In `dobot_pick_place.py`:

- single cube + goal insertion: `dobot_pick_place.py:98-146`

This matters for lab handoff:

- if the lab only wants the robot and viewer, the static XML path is enough
- if they want the task envs, they need the Python env code too

## Where The Scripted Motion Logic Lives

### `demo_pick_place.py`

This is the simplest deterministic demo:

- hard-coded joint targets: `demo_pick_place.py:10-14`
- phase execution loop: `demo_pick_place.py:17-32`
- phase list: `demo_pick_place.py:44-51`

This file is easy to understand and good for a "show the robot doing something" lab handoff.

### `motion_template.py`

This is the best "editable student script" version.

It provides:

- a lightweight API object: `motion_template.py:47-89`
- simulator creation: `motion_template.py:92-104`
- Cartesian helper / IK-ish path: `motion_template.py:119-139`
- `move_to_xyz(...)`: `motion_template.py:154-168`
- direct joint move helper: `motion_template.py:171-189`
- suction helpers: `motion_template.py:202-217`

If the lab wants "a Python file they can modify to try motions", this is the best starting point.

## Where RL Lives

### `train.py`

This is the training driver.

Major responsibilities:

- stage/curriculum definition: `train.py:126-173`
- vectorized env creation: `train.py:196-215`
- PPO / CrossQ model creation: `train.py:218-250`
- CLI config: `train.py:276-495`
- main train loop: `train.py:498-590`

### `record_rollouts.py`

Thin CLI wrapper around replay/export:

- parse args: `record_rollouts.py:8-124`
- call recorder: `record_rollouts.py:127-152`

### `dobot_mujoco/rollout_recording.py`

Used to inspect trained policies after the fact.

Major responsibilities:

- episode summary schema: `rollout_recording.py:15-32`
- per-episode runner: `rollout_recording.py:116-193`
- episode selection logic: `rollout_recording.py:218-252`
- GIF export: `rollout_recording.py:259-263`
- full replay bundle generation: `rollout_recording.py:374-495`

This is RL-adjacent analysis tooling, not required for the simulator itself.

## Files That Are Not Core To The Simulator

These can be removed without breaking the base simulation/model itself:

- `train.py`
- `record_rollouts.py`
- `dobot_mujoco/rollout_recording.py`
- `benchmark_cpu_gpu.slurm`
- `logs/`
- `models/`
- `recordings/`
- `kk_behavioral_shaping_20260423_161213.md`

These are also not required for the simulator runtime:

- `dobot_firm_utils/`
  - hardware-side utilities
- `typings/`
  - type stubs only

## Recommended Cut-Down Versions

### Option A: Viewer-only lab handoff

Keep:

- `main.py`
- `dobot_mujoco/env/assets/dobot_table_scene.xml`
- `dobot_mujoco/env/assets/dobot_suction_cup.xml`
- `dobot_mujoco/env/assets/meshes/`
- a minimal dependency file with `mujoco`

This version is enough to:

- open the viewer
- inspect the robot model
- step the sim

### Option B: Scripted simulator handoff

Keep:

- everything in Option A
- `dobot_mujoco/env/base_env.py`
- `dobot_mujoco/env/dobot_cube_stack.py`
- `dobot_mujoco/env/dobot_pick_place.py`
- `dobot_mujoco/env/__init__.py`
- `dobot_mujoco/__init__.py`
- `demo_pick_place.py`
- `motion_template.py`

Dependencies:

- `mujoco`
- `numpy`
- `gymnasium`
- `gymnasium-robotics`

This version is enough to:

- instantiate the Gym envs
- run scripted demos
- expose the student-friendly API
- keep the small IK helper in `motion_template.py`

### Option C: Full research/RL version

Keep everything in Option B plus:

- `train.py`
- `record_rollouts.py`
- `dobot_mujoco/rollout_recording.py`
- `pyproject.toml`
- `requirements.txt`

Additional dependencies:

- `stable-baselines3`
- `sb3-contrib`
- `imageio`
- `torch`

## If You Want To Strip This Repo To "Simulation Only"

My practical recommendation is:

1. Keep the `dobot_mujoco/` package and assets intact.
2. Keep `main.py`.
3. Keep `motion_template.py` if you want an approachable scripting entrypoint.
4. Delete RL and experiment-output files.
5. Delete `dobot_firm_utils/` unless the lab also needs hardware-side references.

Concretely, a simulation-only repo could look like:

```text
dobot_mujoco/
  __init__.py
  env/
    __init__.py
    base_env.py
    dobot_cube_stack.py
    dobot_pick_place.py
    assets/
      dobot_table_scene.xml
      dobot_suction_cup.xml
      meshes/
main.py
motion_template.py
demo_pick_place.py
README.md
pyproject.toml
```

Then simplify `pyproject.toml` to remove RL-only dependencies if desired.

## Minimal Dependency Ideas

### For `main.py` only

You likely only need:

- `mujoco`

### For env + scripted sim

You need at least:

- `mujoco`
- `numpy`
- `gymnasium`
- `gymnasium-robotics`

### For RL / replay analysis

You additionally need:

- `stable-baselines3`
- `sb3-contrib`
- `torch`
- `imageio`

Current full dependency list is in `pyproject.toml:1-40`.

## Recommended Files To Read First

If you only read a few files, I would read them in this order:

1. `main.py`
   - shows the absolute minimum raw MuJoCo path
2. `dobot_mujoco/env/assets/dobot_table_scene.xml`
   - shows the scene/world composition
3. `dobot_mujoco/env/assets/dobot_suction_cup.xml`
   - shows joints, end effector, actuators
4. `dobot_mujoco/env/dobot_cube_stack.py`
   - shows how actions map into control and how task objects are inserted
5. `dobot_mujoco/env/dobot_pick_place.py`
   - shows the actual single-cube task and reward
6. `motion_template.py`
   - shows the most human-readable scripting surface, including the only IK helper

## Bottom Line

If you are planning a lab handoff:

- there is no big hidden IK subsystem to preserve
- the only IK-ish code is the small Jacobian helper in `motion_template.py`
- the real heart of the simulator is:
  - the two XML files
  - the meshes
  - `dobot_cube_stack.py`
  - `dobot_pick_place.py`
  - `base_env.py`

If you want, the next useful step would be for me to make a second pass and actually produce a **cleaned simulation-only branch layout** with the RL pieces removed and the dependencies simplified.
