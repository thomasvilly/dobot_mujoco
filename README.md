# dobot-mujoco-env

A compact MuJoCo environment for a Dobot manipulator and a small training example.

What this repository contains
- `dobot_mujoco/` — the package with the MuJoCo scene and Gym-compatible environment implementations (look in `dobot_mujoco/env`).
- `main.py` — the quickest way to launch the Dobot scene in the MuJoCo viewer.
- `train.py` — a minimal example that creates `DobotCubeStack-v0`, trains a `CrossQ` agent (from `sb3_contrib`), and saves the trained model.

How it works (brief)
- MuJoCo runs the physics simulation using the XML scene in `dobot_mujoco/env/assets`.
- The environment exposes a continuous action space (5 actions: 4 joint controls + suction pump) and a flat observation vector (joint states, end-effector pose and cube states).
- `train.py` uses Gymnasium to create the env, Stable Baselines3 (sb3_contrib) for the agent, and saves models under `models/`.

## Requirements

- Python 3.10 or higher (see `pyproject.toml`).
- MuJoCo (Python bindings), PyTorch, Stable-Baselines3, sb3-contrib, numpy, and related dependencies listed in `requirements.txt` or `pyproject.toml`.

## Quickstart — create a virtual environment

Run these commands in the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools
```

Option A — install with the pinned/declarative tool used by this project

This repository is configured to be used with `uv` (see `pyproject.toml` for `tool.uv` settings). To sync/install the environment using that configuration run:

```bash
# sync the project environment; choose the extra you want: cpu, cu124
uv sync --extra cpu
# Or with a CUDA compatible device
uv sync --extra cu124
```

Option B — install with pip

```bash
pip install -r requirements.txt
```

## Quick viewer smoke test

To confirm the Dobot model and scene load correctly:

```bash
python main.py
```

This launches the MuJoCo viewer with the Dobot table scene.

If you are using the Python `mujoco` package, it already ships the runtime needed for the viewer. A local MuJoCo installation such as `C:\Users\User\Documents\mujoco-3.7.0-windows-x86_64\` is still useful for native tools like `simulate.exe`, but it is not required just to run `main.py` or the Python environment code.

## Using the training script (`train.py`)

`train.py` is a minimal example showing how to create the environment, train an agent, and optionally export replay bundles for review.

Basic usage:

```bash
python train.py --algo ppo --curriculum basic --n-timesteps 200000 --device cpu
```

Command-line options available (default shown in script):
- `--n-timesteps`: number of training timesteps
- `--curriculum`: `none` or `basic`
- `--n-envs` / `--vec-env`: parallel env count and backend
- `--tensorboard-log`: directory for tensorboard logs
- `--monitor-dir`: directory for Monitor CSV logs
- `--record-rollouts`: after training, save replay GIFs plus JSON/Markdown summaries
- `--progress-bar`: show a progress bar during training

When the script starts it prints detected PyTorch devices and then creates the `DobotPickPlace-v0` environment. PPO is the default algorithm.

Trained models are saved under `models/`. If you pass `--record-rollouts`, curated replay bundles are written under `recordings/`, including:
- top reward episodes
- high reward failures
- random sample episodes
- `report.md` with clickable links to each GIF/JSON summary

Example:

```bash
python train.py \
  --algo ppo \
  --curriculum basic \
  --n-timesteps 200000 \
  --n-envs 8 \
  --vec-env subproc \
  --episode-steps 200 \
  --rollout-steps 128 \
  --batch-size 256 \
  --device cpu \
  --record-rollouts
```

If you already have a checkpoint and just want the review bundle:

```bash
python record_rollouts.py --model-path models/ppo_dobotpickplace_v0.zip
```

## Project structure (important files)

- `dobot_mujoco/` — package containing the environment and assets
  - `dobot_mujoco/env/` — MuJoCo env implementations and XML assets
- `train.py` — small training example that uses Gymnasium + SB3
- `pyproject.toml` — packaging + `uv` configuration and optional extras (cpu/cu128/rocm)
- `requirements.txt` — pip-compatible list of dependencies

## License

See `LICENSE` in the repository root.
