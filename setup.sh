#!/bin/bash
# Build the Crafter training environment on the UPB grid cluster.
#
# Run this INSIDE an interactive job, never on fep -- it compiles wheels.
#
#     srun --partition=ucsx --gres=gpu:1 --cpus-per-task=8 --mem=16G \
#          --time=01:00:00 --pty bash
#     cd /export/projects/grai/$USER/code/crafter-rl
#     bash setup.sh
#
# Idempotent: re-running it reuses the existing env and just re-checks.
#
# Env vars you can override:
#   PROJ      project dir           (default /export/projects/grai)
#   ENV_DIR   conda env prefix      (default $PROJ/$USER/envs/crafter)
#   PY_VER    python version        (default 3.11)
#   CUDA_TAG  torch wheel index tag (default cu121)

set -euo pipefail

PROJ="${PROJ:-/export/projects/grai}"
ENV_DIR="${ENV_DIR:-$PROJ/$USER/envs/crafter}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu121}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "repo     $REPO"
echo "env      $ENV_DIR"
echo "python   $PY_VER"
echo "torch    $CUDA_TAG"
echo

# ---------------------------------------------------------------- 1. sanity

if [[ "$(hostname)" == fep* ]]; then
  echo "ERROR: you are on fep. Get an interactive job first:"
  echo "  srun --partition=ucsx --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=01:00:00 --pty bash"
  exit 1
fi

if [ ! -f "$REPO/setup.py" ] || [ ! -d "$REPO/crafter" ]; then
  echo "ERROR: $REPO does not look like the crafter-rl repo."
  exit 1
fi

mkdir -p "$PROJ/$USER"/{logs,envs}

# ---------------------------------------------------------------- 2. modules

# Name varies between cluster images; take the first CUDA module that exists.
if command -v module >/dev/null 2>&1; then
  for m in libraries/cuda-13.0 libraries/cuda-12.1 libraries/cuda-11.8 cuda; do
    if module load "$m" 2>/dev/null; then
      echo "loaded module $m"
      break
    fi
  done
fi

# ---------------------------------------------------------------- 3. conda

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Try 'module load python/3.11-anaconda' or similar,"
  echo "then check 'module avail' for the anaconda module on this cluster."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
  echo "creating env at $ENV_DIR"
  conda create -p "$ENV_DIR" "python=$PY_VER" -y
else
  echo "env already exists, reusing it"
fi

conda activate "$ENV_DIR"
echo "python: $(which python) -- $(python --version)"

# ---------------------------------------------------------------- 4. packages

python -m pip install --upgrade pip

# torch first, from the CUDA index. A CPU wheel here silently wastes the GPU.
python -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# crafter itself, editable, plus its declared deps (numpy, imageio, pillow,
# opensimplex, ruamel.yaml).
python -m pip install -e "$REPO"

# What the stages and analysis scripts import beyond crafter's own deps.
python -m pip install matplotlib pandas tqdm

# NumPy 2 is fine; the stages were developed on it. Pinned only if you hit the
# vendored analysis/common.py assert -- common/metrics.py works around it.

# ---------------------------------------------------------------- 5. verify

echo
echo "=============== verification ==============="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi (CPU node?)"

python - <<'PY'
import sys
print('python  ', sys.version.split()[0])

import numpy, torch
print('numpy   ', numpy.__version__)
print('torch   ', torch.__version__, '| cuda build:', torch.version.cuda)
print('cuda ok ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu     ', torch.cuda.get_device_name(0))

import crafter
print('crafter ', crafter.__version__ if hasattr(crafter, '__version__') else 'ok')

# The env must actually build and step: catches missing assets/data.yaml.
from common.env import make_env, OBS_SHAPE, N_ACTIONS
env = make_env(seed=0, logdir=None)
obs = env.reset()
obs, reward, done, info = env.step(0)
assert obs.shape == OBS_SHAPE, obs.shape
assert 'discount' in info
print('env     ', obs.shape, N_ACTIONS, 'actions, step ok')

import matplotlib; print('mpl     ', matplotlib.__version__)
PY

# Scorer gate: proves our metric reproduces the published baseline numbers.
echo
echo "--- scorer validation ---"
cd "$REPO" && python -m common.validate_scorer

echo
echo "=============== setup done ==============="
echo "Activate it in every job script with:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_DIR"
