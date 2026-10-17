#!/bin/bash
# Build the Crafter training environment with venv + pip, for cluster nodes
# that have no conda. Same job as setup.sh, without the conda dependency.
#
# Run this INSIDE an interactive job, never on fep -- it compiles wheels.
#
#     srun --partition=ucsx --gres=gpu:1 --cpus-per-task=8 --mem=16G \
#          --time=01:00:00 --pty bash
#     cd ~/code/crafter-rl
#     bash setup_cluster.sh
#
# Idempotent: re-running reuses the existing venv and just re-checks.
#
# Env vars you can override:
#   ENV_DIR   venv location          (default $HOME/envs/crafter)
#   PYTHON    interpreter to build from (default python3)
#   CUDA_TAG  torch wheel index tag  (default cu121)
#   CUDA_MOD  cuda module to load    (default: probe a list)

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/crafter}"
PYTHON="${PYTHON:-python3}"
CUDA_TAG="${CUDA_TAG:-cu121}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "repo     $REPO"
echo "env      $ENV_DIR"
echo "python   $PYTHON"
echo "torch    $CUDA_TAG"
echo

# ---------------------------------------------------------------- 1. sanity

if [[ "$(hostname)" == fep* ]]; then
  echo "ERROR: you are on fep, the front-end. Get a compute node first:"
  echo "  srun --partition=ucsx --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=01:00:00 --pty bash"
  exit 1
fi

if [ ! -f "$REPO/setup.py" ] || [ ! -d "$REPO/crafter" ]; then
  echo "ERROR: $REPO does not look like the crafter-rl repo."
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not found. Pass one: PYTHON=/usr/bin/python3.11 bash $0"
  exit 1
fi

# torch needs 3.9+; the venv is pointless if the interpreter is older.
"$PYTHON" - <<'PY' || exit 1
import sys
if sys.version_info < (3, 9):
    print(f'ERROR: python {sys.version.split()[0]} is too old for current torch '
          '(need 3.9+). Set PYTHON= to a newer interpreter.')
    sys.exit(1)
PY

# ---------------------------------------------------------------- 2. modules

# Only for the system CUDA toolkit. The torch wheel bundles its own runtime, so
# a miss here is not fatal -- torch.cuda.is_available() in section 5 is the real
# check. Names vary per node image; see 'module avail'.
if command -v module >/dev/null 2>&1; then
  for m in ${CUDA_MOD:-libraries/cuda-13 libraries/cuda-13.1 libraries/cuda-12.1 libraries/cuda-11.8 cuda}; do
    if module load "$m" 2>/dev/null; then
      echo "loaded module $m"
      break
    fi
  done
fi

# ---------------------------------------------------------------- 3. venv

mkdir -p "$(dirname "$ENV_DIR")"

if [ ! -d "$ENV_DIR" ]; then
  echo "creating venv at $ENV_DIR"
  "$PYTHON" -m venv "$ENV_DIR"
else
  echo "venv already exists, reusing it"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
echo "python: $(command -v python) -- $(python --version)"

# ---------------------------------------------------------------- 4. packages

python -m pip install --upgrade pip

# torch first, from the CUDA index. A CPU wheel here silently wastes the GPU.
python -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# crafter itself, editable, plus its declared deps (numpy, imageio, pillow,
# opensimplex, ruamel.yaml).
python -m pip install -e "$REPO"

# What the stages and analysis scripts import beyond crafter's own deps.
python -m pip install matplotlib pandas tqdm

# ---------------------------------------------------------------- 5. verify

echo
echo "=============== verification ==============="

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || echo "no nvidia-smi (CPU node?)"

cd "$REPO"
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
python -m common.validate_scorer

# Training on CPU by accident costs a full allocation, so say it loudly.
echo
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU OK"
else
  echo "WARNING: torch.cuda.is_available() is False -- you would train on CPU."
  echo "  - no GPU on this node?  check nvidia-smi above"
  echo "  - wrong wheel?          re-run with CUDA_TAG=cu118 (or a newer tag)"
fi

echo
echo "=============== setup done ==============="
echo "disk: $(du -sh "$ENV_DIR" 2>/dev/null | cut -f1) in $ENV_DIR"
echo
echo "Activate it in every new shell and in jobs/train.sh with:"
echo "  source $ENV_DIR/bin/activate"
