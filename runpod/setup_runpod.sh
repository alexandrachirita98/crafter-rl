#!/bin/bash
# Build the Crafter training environment on a RunPod pod: venv + pip, no conda.
#
# RunPod's pytorch images already ship a CUDA torch, so by default we keep it
# and only add what the stages need. That saves ~3 GB of download, which on a
# billed-by-the-hour box is real money.
#
#     cd /workspace/crafter-rl
#     bash runpod/setup_runpod.sh
#
# Idempotent: re-running reuses the venv and just re-checks.
#
# Env vars you can override:
#   PROJ        persistent volume      (default /workspace)
#   ENV_DIR     venv location          (default $PROJ/envs/crafter)
#   PYTHON      interpreter to build from (default python3)
#   CUDA_TAG    torch wheel index tag  (default cu121; only used if we install torch)
#   FORCE_TORCH 1 = install torch from the CUDA index even if one is present

set -euo pipefail

PROJ="${PROJ:-/workspace}"
ENV_DIR="${ENV_DIR:-$PROJ/envs/crafter}"
PYTHON="${PYTHON:-python3}"
CUDA_TAG="${CUDA_TAG:-cu121}"
FORCE_TORCH="${FORCE_TORCH:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "repo     $REPO"
echo "volume   $PROJ"
echo "env      $ENV_DIR"
echo "python   $PYTHON"
echo

# ---------------------------------------------------------------- 1. sanity

if [ ! -f "$REPO/setup.py" ] || [ ! -d "$REPO/crafter" ]; then
  echo "ERROR: $REPO does not look like the crafter-rl repo."
  exit 1
fi

# The one RunPod-specific trap: a pod's container disk is wiped on stop, only
# the volume survives. A repo outside it means losing results to a restart.
case "$REPO" in
  "$PROJ"/*) ;;
  *) echo "WARNING: the repo is at $REPO, outside the persistent volume $PROJ."
     echo "  A pod stop/restart can wipe it. Move it to $PROJ/crafter-rl first,"
     echo "  or set PROJ= to whatever your persistent mount actually is."
     echo;;
esac

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not found. Pass one: PYTHON=/usr/bin/python3.11 bash $0"
  exit 1
fi

"$PYTHON" - <<'PY' || exit 1
import sys
if sys.version_info < (3, 9):
    print(f'ERROR: python {sys.version.split()[0]} is too old for current torch '
          '(need 3.9+). Set PYTHON= to a newer interpreter.')
    sys.exit(1)
PY

# ---------------------------------------------------------------- 2. venv

mkdir -p "$(dirname "$ENV_DIR")"

# --system-site-packages so the image's own CUDA torch stays visible. Building
# a sealed venv is what makes people re-download 3 GB of wheels.
if [ ! -d "$ENV_DIR" ]; then
  echo "creating venv at $ENV_DIR"
  "$PYTHON" -m venv --system-site-packages "$ENV_DIR"
else
  echo "venv already exists, reusing it"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
echo "python: $(command -v python) -- $(python --version)"

# ---------------------------------------------------------------- 3. packages

python -m pip install --upgrade pip

HAVE_CUDA_TORCH=0
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  HAVE_CUDA_TORCH=1
fi

if [ "$HAVE_CUDA_TORCH" -eq 1 ] && [ "$FORCE_TORCH" -eq 0 ]; then
  echo "using the image's torch: $(python -c 'import torch; print(torch.__version__)')"
else
  echo "installing torch from the $CUDA_TAG index"
  python -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# crafter itself, editable, plus its declared deps.
python -m pip install -e "$REPO"

# What the stages and analysis scripts import beyond crafter's own deps.
python -m pip install matplotlib pandas tqdm

# ---------------------------------------------------------------- 4. verify

echo
echo "=============== verification ==============="

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || echo "no nvidia-smi -- is a GPU attached to this pod?"

nproc | sed 's/^/cores   /'
free -g | awk '/^Mem:/ {print "ram     " $2 " GB"}'
df -h "$PROJ" | tail -1 | awk '{print "volume  " $4 " free of " $2}'

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
print('crafter ', getattr(crafter, '__version__', 'ok'))

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

# Stage 12 replays recorded demos, and demos/ is gitignored -- a fresh clone
# has none, and the stage would train without them and look like it failed.
echo
if [ -n "$(ls "$REPO"/demos/*.npz 2>/dev/null)" ]; then
  echo "demos   $(ls "$REPO"/demos/*.npz | wc -l) episodes in demos/ (stage 12 ok)"
else
  echo "NOTE: demos/ is empty or missing, and it is gitignored so a clone has none."
  echo "  Stage 12 needs it. Either push them from your laptop:"
  echo "    bash runpod/sync.sh push-demos <ssh-port> <pod-host>"
  echo "  or regenerate on the pod:  python record_demos.py"
fi

echo
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU OK"
else
  echo "WARNING: torch.cuda.is_available() is False -- you would train on CPU."
  echo "  - no GPU on this pod?   check nvidia-smi above"
  echo "  - wrong wheel?          FORCE_TORCH=1 CUDA_TAG=cu118 bash $0"
fi

echo
echo "=============== setup done ==============="
echo "disk: $(du -sh "$ENV_DIR" 2>/dev/null | cut -f1) in $ENV_DIR"
echo
echo "Activate it in every new shell, and in any nohup'd run:"
echo "  source $ENV_DIR/bin/activate"
