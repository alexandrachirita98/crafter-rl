#!/bin/bash
#SBATCH --job-name=crafter
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:1
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

echo "node=$(hostname) job=$SLURM_JOB_ID seed=$SLURM_ARRAY_TASK_ID"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"
df -h "$HOME" | tail -1

# Raw logdir on node-local scratch; run.sh copies out the small files.
./run.sh "${STAGE:?set STAGE}" \
  --steps "${STEPS:-200000}" \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --name "${NAME:-live_run}" \
  --logdir "${TMPDIR:-/tmp}/crafter-$SLURM_JOB_ID"
