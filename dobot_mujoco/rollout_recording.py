from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class EpisodeSummary:
    episode_index: int
    seed: int
    total_reward: float
    length: int
    success: bool
    ever_grasped: bool
    ever_lifted: bool
    max_lift_fraction: float
    max_cube_height_above_table: float
    min_ee_to_cube_distance: float | None
    min_cube_to_goal_distance: float | None
    final_ee_to_cube_distance: float | None
    final_cube_to_goal_distance: float | None
    final_lift_fraction: float | None
    reward_per_step: float
    final_info: dict[str, Any]


def _normalize_model_path(model_path: str | os.PathLike[str]) -> Path:
    path = Path(model_path)
    if path.exists():
        return path

    zipped_path = Path(f"{path}.zip")
    if zipped_path.exists():
        return zipped_path

    raise FileNotFoundError(f"Could not find trained model at {model_path}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def summary_to_dict(summary: EpisodeSummary) -> dict[str, Any]:
    return _to_jsonable(asdict(summary))


def _load_model(algo: str, model_path: str | os.PathLike[str]):
    normalized_path = str(_normalize_model_path(model_path))
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(normalized_path)

    from sb3_contrib import CrossQ

    return CrossQ.load(normalized_path)


def _make_env(
    env_id: str,
    episode_steps: int,
    env_kwargs: dict[str, Any],
    render_mode: str | None,
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


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _seed_policy_sampling(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_policy_episode(
    model,
    env,
    seed: int,
    episode_index: int = 0,
    deterministic: bool = True,
    capture_frames: bool = False,
    frame_skip: int = 4,
) -> tuple[EpisodeSummary, list[np.ndarray]]:
    frame_skip = max(1, frame_skip)
    frames: list[np.ndarray] = []
    obs, info = env.reset(seed=seed)
    last_info = dict(info)

    if not deterministic:
        _seed_policy_sampling(seed)

    ever_grasped = bool(info.get("grasped", False))
    ever_lifted = bool(info.get("is_lifted", False))
    max_lift_fraction = float(info.get("lift_fraction", 0.0))
    max_cube_height_above_table = float(info.get("cube_height_above_table", 0.0))
    min_ee_to_cube_distance = float(info.get("ee_to_cube_distance", np.inf))
    min_cube_to_goal_distance = float(info.get("cube_to_goal_distance", np.inf))

    if capture_frames:
        frames.append(np.asarray(env.render(), dtype=np.uint8))

    total_reward = 0.0
    length = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        total_reward += float(reward)
        length += 1
        done = terminated or truncated

        ever_grasped = ever_grasped or bool(info.get("grasped", False))
        ever_lifted = ever_lifted or bool(info.get("is_lifted", False))
        max_lift_fraction = max(max_lift_fraction, float(info.get("lift_fraction", 0.0)))
        max_cube_height_above_table = max(
            max_cube_height_above_table,
            float(info.get("cube_height_above_table", 0.0)),
        )
        min_ee_to_cube_distance = min(
            min_ee_to_cube_distance,
            float(info.get("ee_to_cube_distance", np.inf)),
        )
        min_cube_to_goal_distance = min(
            min_cube_to_goal_distance,
            float(info.get("cube_to_goal_distance", np.inf)),
        )

        if capture_frames and ((length % frame_skip == 0) or done):
            frames.append(np.asarray(env.render(), dtype=np.uint8))

    summary = EpisodeSummary(
        episode_index=episode_index,
        seed=seed,
        total_reward=total_reward,
        length=length,
        success=bool(last_info.get("is_success", False)),
        ever_grasped=ever_grasped,
        ever_lifted=ever_lifted,
        max_lift_fraction=max_lift_fraction,
        max_cube_height_above_table=max_cube_height_above_table,
        min_ee_to_cube_distance=_finite_or_none(min_ee_to_cube_distance),
        min_cube_to_goal_distance=_finite_or_none(min_cube_to_goal_distance),
        final_ee_to_cube_distance=_finite_or_none(last_info.get("ee_to_cube_distance", np.inf)),
        final_cube_to_goal_distance=_finite_or_none(
            last_info.get("cube_to_goal_distance", np.inf)
        ),
        final_lift_fraction=_finite_or_none(last_info.get("lift_fraction", np.inf)),
        reward_per_step=(total_reward / max(length, 1)),
        final_info=_to_jsonable(last_info),
    )
    return summary, frames


def evaluate_policy_episodes(
    model,
    env,
    n_episodes: int,
    base_seed: int,
    deterministic: bool = True,
) -> list[EpisodeSummary]:
    summaries: list[EpisodeSummary] = []
    for episode_index in range(n_episodes):
        seed = base_seed + episode_index
        summary, _ = run_policy_episode(
            model=model,
            env=env,
            seed=seed,
            episode_index=episode_index,
            deterministic=deterministic,
            capture_frames=False,
        )
        summaries.append(summary)
    return summaries


def select_episode_groups(
    summaries: list[EpisodeSummary],
    top_k: int,
    random_k: int,
    failure_k: int,
    selection_seed: int,
) -> dict[str, list[EpisodeSummary]]:
    sorted_by_reward = sorted(summaries, key=lambda item: item.total_reward, reverse=True)
    top_reward = sorted_by_reward[: min(top_k, len(sorted_by_reward))]

    used_seeds = {item.seed for item in top_reward}
    failure_candidates = [
        item
        for item in sorted_by_reward
        if (not item.success) and item.seed not in used_seeds
    ]
    high_reward_failures = failure_candidates[: min(failure_k, len(failure_candidates))]
    used_seeds.update(item.seed for item in high_reward_failures)

    random_pool = [item for item in summaries if item.seed not in used_seeds]
    random_sample: list[EpisodeSummary] = []
    if random_pool and random_k > 0:
        rng = np.random.default_rng(selection_seed)
        chosen_indices = rng.choice(
            len(random_pool),
            size=min(random_k, len(random_pool)),
            replace=False,
        )
        random_sample = [random_pool[index] for index in sorted(chosen_indices.tolist())]

    return {
        "top_reward": top_reward,
        "high_reward_failures": high_reward_failures,
        "random_sample": random_sample,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _write_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    duration = 1.0 / max(fps, 1)
    imageio.mimsave(path, frames, duration=duration, loop=0)


def _aggregate_summaries(summaries: list[EpisodeSummary]) -> dict[str, Any]:
    if not summaries:
        return {
            "screened_episodes": 0,
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "median_reward": 0.0,
            "max_reward": 0.0,
            "mean_length": 0.0,
            "grasp_rate": 0.0,
            "lift_rate": 0.0,
            "mean_max_lift_fraction": 0.0,
        }

    rewards = np.array([summary.total_reward for summary in summaries], dtype=np.float64)
    lengths = np.array([summary.length for summary in summaries], dtype=np.float64)
    successes = np.array([summary.success for summary in summaries], dtype=np.float64)
    grasped = np.array([summary.ever_grasped for summary in summaries], dtype=np.float64)
    lifted = np.array([summary.ever_lifted for summary in summaries], dtype=np.float64)
    max_lift_fraction = np.array(
        [summary.max_lift_fraction for summary in summaries],
        dtype=np.float64,
    )

    failure_summaries = [summary for summary in summaries if not summary.success]
    success_summaries = [summary for summary in summaries if summary.success]

    aggregate = {
        "screened_episodes": int(len(summaries)),
        "success_rate": float(successes.mean()),
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "max_reward": float(rewards.max()),
        "mean_length": float(lengths.mean()),
        "grasp_rate": float(grasped.mean()),
        "lift_rate": float(lifted.mean()),
        "mean_max_lift_fraction": float(max_lift_fraction.mean()),
    }
    if success_summaries:
        aggregate["best_success_reward"] = float(
            max(summary.total_reward for summary in success_summaries)
        )
    if failure_summaries:
        aggregate["best_failure_reward"] = float(
            max(summary.total_reward for summary in failure_summaries)
        )
    return aggregate


def _group_title(group_name: str) -> str:
    return group_name.replace("_", " ").title()


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _build_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Rollout Review",
        "",
        f"- Model: `{manifest['model_path']}`",
        f"- Algorithm: `{manifest['algo']}`",
        f"- Generated: `{manifest['created_at']}`",
        f"- Episodes screened: `{manifest['aggregate']['screened_episodes']}`",
        f"- Success rate: `{manifest['aggregate']['success_rate']:.3f}`",
        f"- Mean reward: `{manifest['aggregate']['mean_reward']:.2f}`",
        f"- Grasp rate: `{manifest['aggregate']['grasp_rate']:.3f}`",
        f"- Lift rate: `{manifest['aggregate']['lift_rate']:.3f}`",
        "",
        "## Reward Hack Checks",
        "",
        f"- Best failure reward: `{manifest['aggregate'].get('best_failure_reward', 0.0):.2f}`",
        f"- Best success reward: `{manifest['aggregate'].get('best_success_reward', 0.0):.2f}`",
        "",
    ]

    for group_name, entries in manifest["groups"].items():
        lines.append(f"## {_group_title(group_name)}")
        lines.append("")
        if not entries:
            lines.append("No episodes selected for this group.")
            lines.append("")
            continue

        lines.append(
            "| Episode | Reward | Success | Grasped | Lifted | Min goal dist | Video | Summary |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for entry in entries:
            lines.append(
                "| "
                f"{entry['episode_index']} | "
                f"{entry['total_reward']:.2f} | "
                f"{int(entry['success'])} | "
                f"{int(entry['ever_grasped'])} | "
                f"{int(entry['ever_lifted'])} | "
                f"{_format_float(entry['min_cube_to_goal_distance'])} | "
                f"[gif]({entry['video_path']}) | "
                f"[json]({entry['summary_path']}) |"
            )
        lines.append("")

    return "\n".join(lines)


def record_saved_model_rollouts(
    algo: str,
    model_path: str | os.PathLike[str],
    env_id: str,
    episode_steps: int,
    env_kwargs: dict[str, Any],
    output_root: str | os.PathLike[str],
    run_name: str | None = None,
    rollout_count: int = 64,
    top_k: int = 10,
    random_k: int = 10,
    failure_k: int = 10,
    base_seed: int = 50_000,
    deterministic: bool = True,
    width: int = 360,
    height: int = 360,
    frame_skip: int = 4,
    fps: int = 10,
) -> Path:
    model_path = _normalize_model_path(model_path)
    model = _load_model(algo, model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = run_name or model_path.stem
    output_dir = Path(output_root) / f"{run_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_env = _make_env(
        env_id=env_id,
        episode_steps=episode_steps,
        env_kwargs=env_kwargs,
        render_mode=None,
    )
    summaries = evaluate_policy_episodes(
        model=model,
        env=eval_env,
        n_episodes=rollout_count,
        base_seed=base_seed,
        deterministic=deterministic,
    )
    eval_env.close()

    selection = select_episode_groups(
        summaries=summaries,
        top_k=top_k,
        random_k=random_k,
        failure_k=failure_k,
        selection_seed=base_seed + 1_000_000,
    )

    record_env = _make_env(
        env_id=env_id,
        episode_steps=episode_steps,
        env_kwargs=env_kwargs,
        render_mode="rgb_array",
        width=width,
        height=height,
    )

    group_entries: dict[str, list[dict[str, Any]]] = {}
    for group_name, selected_summaries in selection.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_entries[group_name] = []

        for item_index, selected_summary in enumerate(selected_summaries):
            replay_summary, frames = run_policy_episode(
                model=model,
                env=record_env,
                seed=selected_summary.seed,
                episode_index=selected_summary.episode_index,
                deterministic=deterministic,
                capture_frames=True,
                frame_skip=frame_skip,
            )
            stem = f"episode_{item_index:02d}_seed_{selected_summary.seed}"
            video_path = group_dir / f"{stem}.gif"
            summary_path = group_dir / f"{stem}.json"
            _write_gif(video_path, frames, fps=fps)
            _write_json(summary_path, summary_to_dict(replay_summary))

            group_entries[group_name].append(
                {
                    **summary_to_dict(replay_summary),
                    "video_path": os.path.relpath(video_path, output_dir),
                    "summary_path": os.path.relpath(summary_path, output_dir),
                }
            )

    record_env.close()

    manifest = {
        "algo": algo,
        "model_path": str(model_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_id": env_id,
        "episode_steps": episode_steps,
        "env_kwargs": _to_jsonable(env_kwargs),
        "selection": {
            "rollout_count": rollout_count,
            "top_k": top_k,
            "random_k": random_k,
            "failure_k": failure_k,
            "base_seed": base_seed,
            "deterministic": deterministic,
            "width": width,
            "height": height,
            "frame_skip": frame_skip,
            "fps": fps,
        },
        "aggregate": _aggregate_summaries(summaries),
        "all_episodes": [summary_to_dict(summary) for summary in summaries],
        "groups": group_entries,
    }

    manifest_path = output_dir / "manifest.json"
    all_episodes_path = output_dir / "all_episodes.json"
    report_path = output_dir / "report.md"
    _write_json(manifest_path, manifest)
    _write_json(all_episodes_path, manifest["all_episodes"])
    report_path.write_text(_build_report(manifest), encoding="utf-8")
    return output_dir
