#!/bin/bash
#SBATCH --job-name=dobot-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log

set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1

REPO_ROOT="$HOME/dobot_mujoco"
COMMON_ARGS=(
  --algo ppo
  --curriculum basic
  --n-timesteps 8192
  --n-envs 8
  --vec-env subproc
  --episode-steps 200
  --rollout-steps 128
  --batch-size 256
  --eval-freq 2048
  --n-eval-episodes 4
)

cd "$REPO_ROOT"
. .venv/bin/activate

python - <<'PY'
import os
import platform

import torch

print("hostname", platform.node())
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}", torch.cuda.get_device_name(index))
print("mujoco_gl", os.environ.get("MUJOCO_GL"))
print("pyopengl_platform", os.environ.get("PYOPENGL_PLATFORM"))
PY

run_case() {
  local name="$1"
  local device="$2"
  local outdir="$REPO_ROOT/benchmarks/${SLURM_JOB_ID}/${name}"

  mkdir -p "$outdir"
  echo
  echo "=== ${name^^} benchmark (${device}) ==="
  (
    cd "$outdir"
    PYTHONPATH="$REPO_ROOT" /usr/bin/time \
      -o time.txt \
      -f "elapsed_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\ncpu_percent=%P\nmax_rss_kb=%M" \
      python "$REPO_ROOT/train.py" "${COMMON_ARGS[@]}" --device "$device" \
      > train.log 2>&1
  )

  echo "--- timing (${name}) ---"
  cat "$outdir/time.txt"
  echo "--- latest eval lines (${name}) ---"
  grep -E '^(\[eval\]|PyTorch CUDA devices:|=== Stage|time/|rollout/|eval/)' "$outdir/train.log" | tail -n 40 || true
}

run_case cpu cpu
run_case gpu cuda

echo
echo "Benchmark artifacts written to: $REPO_ROOT/benchmarks/${SLURM_JOB_ID}"
