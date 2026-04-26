import argparse
import os
from dataclasses import dataclass

import dobot_mujoco  # noqa: F401
import gymnasium as gym
import numpy as np
from dobot_mujoco.rollout_recording import evaluate_policy_episodes, record_saved_model_rollouts


@dataclass
class TrainingStage:
    name: str
    timesteps: int
    episode_steps: int
    env_kwargs: dict


class SuccessEvalCallback:
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        best_model_path: str | None,
        base_seed: int,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.last_eval_timestep = 0
                self.best_success_rate = -np.inf

            def _on_step(self) -> bool:
                if self.outer.eval_freq <= 0:
                    return True

                if self.num_timesteps - self.last_eval_timestep < self.outer.eval_freq:
                    return True

                self.last_eval_timestep = self.num_timesteps
                summaries = evaluate_policy_episodes(
                    model=self.model,
                    env=self.outer.eval_env,
                    n_episodes=self.outer.n_eval_episodes,
                    base_seed=self.outer.base_seed,
                    deterministic=self.outer.deterministic,
                )
                rewards = np.array(
                    [summary.total_reward for summary in summaries],
                    dtype=np.float64,
                )
                lengths = np.array([summary.length for summary in summaries], dtype=np.float64)
                successes = np.array(
                    [summary.success for summary in summaries],
                    dtype=np.float64,
                )
                grasped = np.array(
                    [summary.ever_grasped for summary in summaries],
                    dtype=np.float64,
                )
                lifted = np.array(
                    [summary.ever_lifted for summary in summaries],
                    dtype=np.float64,
                )
                lift_fraction = np.array(
                    [summary.max_lift_fraction for summary in summaries],
                    dtype=np.float64,
                )

                mean_reward = float(rewards.mean())
                mean_length = float(lengths.mean())
                success_rate = float(successes.mean())
                grasp_rate = float(grasped.mean())
                lift_rate = float(lifted.mean())
                mean_max_lift_fraction = float(lift_fraction.mean())
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/mean_ep_length", mean_length)
                self.logger.record("eval/success_rate", success_rate)
                self.logger.record("eval/grasp_rate", grasp_rate)
                self.logger.record("eval/lift_rate", lift_rate)
                self.logger.record("eval/mean_max_lift_fraction", mean_max_lift_fraction)
                self.logger.dump(self.num_timesteps)
                print(
                    f"[eval] steps={self.num_timesteps} "
                    f"reward={mean_reward:.2f} "
                    f"ep_len={mean_length:.1f} "
                    f"success={success_rate:.3f} "
                    f"grasp={grasp_rate:.3f} "
                    f"lift={lift_rate:.3f}"
                )

                if (
                    self.outer.best_model_path is not None
                    and success_rate >= self.best_success_rate
                ):
                    self.best_success_rate = success_rate
                    self.model.save(self.outer.best_model_path)
                    print(
                        f"[eval] saved best model to {self.outer.best_model_path}.zip "
                        f"(success={success_rate:.3f})"
                    )

                return True

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_path = best_model_path
        self.base_seed = base_seed
        self.callback = _Callback(self)


def split_timesteps(total_timesteps: int, fractions: list[float]) -> list[int]:
    raw = [int(total_timesteps * fraction) for fraction in fractions[:-1]]
    allocated = sum(raw)
    raw.append(total_timesteps - allocated)
    return raw


def build_stages(args) -> list[TrainingStage]:
    base_env_kwargs = {
        "position_jitter": args.position_jitter,
        "goal_distance_scale": args.goal_distance_scale,
        "success_tolerance": args.success_tolerance,
    }

    if args.curriculum == "none":
        return [
            TrainingStage(
                name="full",
                timesteps=args.n_timesteps,
                episode_steps=args.episode_steps,
                env_kwargs=base_env_kwargs,
            )
        ]

    stage_timesteps = split_timesteps(args.n_timesteps, [0.2, 0.3, 0.5])
    return [
        TrainingStage(
            name="near_goal",
            timesteps=stage_timesteps[0],
            episode_steps=min(args.episode_steps, 160),
            env_kwargs={
                **base_env_kwargs,
                "position_jitter": 0.0,
                "goal_distance_scale": 0.35,
                "success_tolerance": max(args.success_tolerance, 0.05),
            },
        ),
        TrainingStage(
            name="mid_goal",
            timesteps=stage_timesteps[1],
            episode_steps=min(args.episode_steps, 180),
            env_kwargs={
                **base_env_kwargs,
                "position_jitter": min(args.position_jitter, 0.01),
                "goal_distance_scale": 0.65,
                "success_tolerance": max(args.success_tolerance, 0.04),
            },
        ),
        TrainingStage(
            name="full_goal",
            timesteps=stage_timesteps[2],
            episode_steps=args.episode_steps,
            env_kwargs=base_env_kwargs,
        ),
    ]


def make_single_env(
    env_id: str,
    episode_steps: int,
    env_kwargs: dict,
    render_mode: str | None = None,
    width: int | None = None,
    height: int | None = None,
):
    make_kwargs = {
        "max_episode_steps": episode_steps,
        "render_mode": render_mode,
        **env_kwargs,
    }
    if width is not None:
        make_kwargs["width"] = width
    if height is not None:
        make_kwargs["height"] = height
    return gym.make(env_id, **make_kwargs)


def make_training_env_compat(args, stage: TrainingStage):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    def make_env(rank: int):
        def _init():
            env = make_single_env(args.env_id, stage.episode_steps, stage.env_kwargs)
            filename = None
            if args.monitor_dir is not None:
                os.makedirs(args.monitor_dir, exist_ok=True)
                filename = os.path.join(args.monitor_dir, f"{stage.name}_env{rank}")
            env = Monitor(env, filename=filename, info_keywords=("is_success",))
            env.reset(seed=args.seed + rank)
            return env

        return _init

    return vec_env_cls([make_env(rank) for rank in range(args.n_envs)])


def build_model(args, env):
    if args.algo == "ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            device=args.device,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.2,
        )

    from sb3_contrib import CrossQ

    return CrossQ(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        learning_starts=max(args.batch_size * 4, 2_000),
        buffer_size=200_000,
    )


def print_torch_devices() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            devices = [
                torch.cuda.get_device_name(device)
                for device in range(torch.cuda.device_count())
            ]
            print(f"PyTorch CUDA devices: {devices}")
        else:
            print("PyTorch CUDA devices: []")
    except ImportError:
        print("PyTorch is not installed yet.")


def build_model_name(args) -> str:
    model_name = f"{args.algo.lower()}_{args.env_id.replace('-', '_').lower()}"
    if args.run_name:
        model_name = f"{model_name}_{args.run_name}"
    return model_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "crossq"],
        help="RL algorithm to train with.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="DobotPickPlace-v0",
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="basic",
        choices=["none", "basic"],
        help="Use a short staged curriculum before the full task.",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=300_000,
        help="Total training timesteps across all curriculum stages.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=200,
        help="Maximum env steps per episode.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
        help="Vectorized environment backend.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=256,
        help="PPO rollout length per environment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for PPO or CrossQ.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Discount factor.",
    )
    parser.add_argument(
        "--position-jitter",
        type=float,
        default=0.0,
        help="Random XY jitter for the pick/place task in metres.",
    )
    parser.add_argument(
        "--goal-distance-scale",
        type=float,
        default=1.0,
        help="Scale the goal distance relative to the full pick/place displacement.",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=0.035,
        help="Goal tolerance in metres.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes per evaluation pass.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Stable-Baselines logging interval.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--monitor-dir",
        type=str,
        default=None,
        help="Directory for Monitor CSV logs.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show progress bar during training.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label appended to saved checkpoints and replay bundles.",
    )
    parser.add_argument(
        "--record-rollouts",
        action="store_true",
        help="After training, save a curated set of replay GIFs and JSON summaries.",
    )
    parser.add_argument(
        "--record-model",
        type=str,
        default="final",
        choices=["final", "best", "both"],
        help="Which trained checkpoint to record after training.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="recordings",
        help="Directory where rollout review bundles are written.",
    )
    parser.add_argument(
        "--record-rollout-count",
        type=int,
        default=64,
        help="How many episodes to screen before selecting replay examples.",
    )
    parser.add_argument(
        "--record-top-k",
        type=int,
        default=10,
        help="Number of highest-reward episodes to save as replay GIFs.",
    )
    parser.add_argument(
        "--record-random-k",
        type=int,
        default=10,
        help="Number of random screened episodes to save as replay GIFs.",
    )
    parser.add_argument(
        "--record-failure-k",
        type=int,
        default=10,
        help="Number of high-reward failures to save for reward-hack inspection.",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=360,
        help="Replay GIF width in pixels.",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=360,
        help="Replay GIF height in pixels.",
    )
    parser.add_argument(
        "--record-frame-skip",
        type=int,
        default=4,
        help="Capture every Nth environment step when writing replay GIFs.",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=10,
        help="Playback FPS for replay GIFs.",
    )
    parser.add_argument(
        "--record-seed-offset",
        type=int,
        default=50_000,
        help="Seed offset used for rollout screening and replay recording.",
    )
    parser.add_argument(
        "--record-stochastic",
        action="store_true",
        help="Use stochastic policy sampling when screening and recording replay episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_torch_devices()

    if args.algo == "ppo" and args.batch_size > args.rollout_steps * args.n_envs:
        raise ValueError(
            "--batch-size must be <= rollout_steps * n_envs for PPO. "
            f"Got batch_size={args.batch_size} and rollout_steps*n_envs={args.rollout_steps * args.n_envs}."
        )

    stages = build_stages(args)
    os.makedirs("models", exist_ok=True)
    model_name = build_model_name(args)
    best_model_path = os.path.join("models", f"best_{model_name}")

    model = None
    for stage_idx, stage in enumerate(stages):
        print(
            f"\n=== Stage {stage_idx + 1}/{len(stages)}: {stage.name} ===\n"
            f"timesteps={stage.timesteps} episode_steps={stage.episode_steps} "
            f"env_kwargs={stage.env_kwargs}"
        )

        train_env = make_training_env_compat(args, stage)
        eval_env = make_single_env(args.env_id, stage.episode_steps, stage.env_kwargs)

        if model is None:
            model = build_model(args, train_env)
        else:
            model.set_env(train_env)

        eval_callback = SuccessEvalCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            best_model_path=best_model_path,
            base_seed=args.seed + 10_000 * (stage_idx + 1),
        )

        model.learn(
            total_timesteps=stage.timesteps,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
            reset_num_timesteps=(stage_idx == 0),
            callback=eval_callback.callback,
        )

        stage_model_path = os.path.join(
            "models",
            f"{model_name}_{stage.name}",
        )
        model.save(stage_model_path)
        print(f"Saved stage checkpoint to {stage_model_path}.zip")

        train_env.close()
        eval_env.close()

    final_model_path = os.path.join("models", model_name)
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}.zip")

    if args.record_rollouts:
        os.makedirs(args.record_dir, exist_ok=True)
        record_targets: list[tuple[str, str]] = []
        if args.record_model in {"final", "both"}:
            record_targets.append(("final", final_model_path))
        if args.record_model in {"best", "both"} and os.path.exists(
            f"{best_model_path}.zip"
        ):
            record_targets.append(("best", best_model_path))

        for label, model_path in record_targets:
            bundle_dir = record_saved_model_rollouts(
                algo=args.algo,
                model_path=model_path,
                env_id=args.env_id,
                episode_steps=stages[-1].episode_steps,
                env_kwargs=stages[-1].env_kwargs,
                output_root=args.record_dir,
                run_name=f"{model_name}_{label}",
                rollout_count=args.record_rollout_count,
                top_k=args.record_top_k,
                random_k=args.record_random_k,
                failure_k=args.record_failure_k,
                base_seed=args.seed + args.record_seed_offset,
                deterministic=not args.record_stochastic,
                width=args.record_width,
                height=args.record_height,
                frame_skip=args.record_frame_skip,
                fps=args.record_fps,
            )
            print(f"Saved rollout review bundle to {bundle_dir}")
