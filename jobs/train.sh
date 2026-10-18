#!/bin/bash
#SBATCH --job-name=crafter
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:1
# Each parallel env is a separate Crafter instance stepped on the CPU, so an
# ENVS=N run wants roughly N cores for the envs plus one for the learner. This
# default covers ENVS up to 8; for ENVS=16 override it at submit time, which
# takes precedence over the directive:
#     sbatch --cpus-per-task=20 jobs/train.sh
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=1-2

set -euo pipefail
PROJ=$HOME

module load libraries/cuda-13 2>/dev/null || true

# A non-interactive shell will not activate the env for you.
source $PROJ/envs/crafter/bin/activate

cd $PROJ/code/crafter-rl

echo "node=$(hostname) job=$SLURM_JOB_ID seed=$SLURM_ARRAY_TASK_ID" \
     "envs=${ENVS:-1} steps=${STEPS:-200000}"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"
df -h "$HOME" | tail -1

# Stage-specific train.py flags, e.g. EXTRA="--eps-decay-steps 50000".
# Unquoted on purpose: it has to split into separate words.
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"

# The envs are the parallel part; torch's own intra-op threads would fight
# them for the same cores. One learner thread per core not spent on an env.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Raw logdir on node-local scratch; run.sh copies out the small files.
./run.sh "${STAGE:?set STAGE}" \
  --steps "${STEPS:-200000}" \
  --envs "${ENVS:-1}" \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --name "${NAME:-live_run}" \
  --logdir "${TMPDIR:-/tmp}/crafter-$SLURM_JOB_ID" \
  ${EXTRA_ARGS[@]+-- "${EXTRA_ARGS[@]}"}
