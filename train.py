import os

import dobot_mujoco
import gymnasium as gym


if __name__ == "__main__":
    import argparse

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
        "--n-timesteps", type=int, default=200_000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--log-interval", type=int, default=2, help="Log interval (in episodes)"
    )
    parser.add_argument(
        "--tensorboard-log", type=str, default=None, help="Tensorboard log directory"
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="Show progress bar during training"
    )
    parser.add_argument(
        "--position-jitter",
        type=float,
        default=0.0,
        help="Random XY jitter for the pick/place task in metres.",
    )
    args = parser.parse_args()

    try:
        import torch

        print(
            f"Pytorch devices available: {[torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())]}"
        )
    except ImportError:
        print("PyTorch is not installed yet.")

    env_kwargs = {}
    if args.env_id == "DobotPickPlace-v0":
        env_kwargs["position_jitter"] = args.position_jitter

    env = gym.make(
        args.env_id,
        max_episode_steps=40 * 90,
        render_mode="rgb_array",
        **env_kwargs,
    )

    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
        )
    else:
        from sb3_contrib import CrossQ

        model = CrossQ("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)

    model.learn(
        total_timesteps=args.n_timesteps,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
    )
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.algo.lower()}_{args.env_id.replace('-', '_').lower()}")
