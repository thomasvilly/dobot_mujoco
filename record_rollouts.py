import argparse

import dobot_mujoco  # noqa: F401

from dobot_mujoco.rollout_recording import record_saved_model_rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "crossq"],
        help="Algorithm class used to load the checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a saved SB3 model (.zip is optional).",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="DobotPickPlace-v0",
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=200,
        help="Maximum env steps per episode.",
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
        "--output-dir",
        type=str,
        default="recordings",
        help="Directory where rollout review bundles are written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label for the rollout bundle directory.",
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=64,
        help="How many episodes to screen before selecting replay examples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-reward episodes to save as replay GIFs.",
    )
    parser.add_argument(
        "--random-k",
        type=int,
        default=10,
        help="Number of random screened episodes to save as replay GIFs.",
    )
    parser.add_argument(
        "--failure-k",
        type=int,
        default=10,
        help="Number of high-reward failures to save for reward-hack inspection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50_000,
        help="Base seed used for rollout screening and replay recording.",
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
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling when screening and recording replay episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle_dir = record_saved_model_rollouts(
        algo=args.algo,
        model_path=args.model_path,
        env_id=args.env_id,
        episode_steps=args.episode_steps,
        env_kwargs={
            "position_jitter": args.position_jitter,
            "goal_distance_scale": args.goal_distance_scale,
            "success_tolerance": args.success_tolerance,
        },
        output_root=args.output_dir,
        run_name=args.run_name,
        rollout_count=args.rollout_count,
        top_k=args.top_k,
        random_k=args.random_k,
        failure_k=args.failure_k,
        base_seed=args.seed,
        deterministic=not args.stochastic,
        width=args.record_width,
        height=args.record_height,
        frame_skip=args.record_frame_skip,
        fps=args.record_fps,
    )
    print(f"Saved rollout review bundle to {bundle_dir}")
