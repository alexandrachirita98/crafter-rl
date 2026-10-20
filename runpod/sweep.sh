#!/bin/bash
# Run a list of stages one after another on a RunPod pod, surviving ssh drops.
#
# This is the RunPod stand-in for sbatch + jobs/train.sh: no queue, so the
# sequencing is a loop and the "detach" is setsid+nohup. One job at a time is
# deliberate -- Crafter steps in Python on the CPU, so two runs on one box
# halve each other's fps and you pay twice the hours for the same work.
#
# Usage:
#     bash runpod/sweep.sh [options] <stage> [<stage> ...]
#
#     bash runpod/sweep.sh --smoke 10c                  # 2k shakedown, foreground
#     bash runpod/sweep.sh --detach 02 03 04            # 200k x 2 seeds, detached
#     bash runpod/sweep.sh --steps 1000000 --seeds 1 --name final --detach 03
#     bash runpod/sweep.sh --contrib --detach           # the three C stages
#     bash runpod/sweep.sh --ladder --detach            # 02..10c, the full stack
#
# Options:
#     --steps N      env steps per seed, total across envs  (default 200000)
#     --envs N       parallel envs    (default: cores - 2, capped at 8)
#     --seeds a,b    seeds per stage  (default 1,2)
#     --name DIR     results subdir   (default live_run)
#     --smoke        2k-step shakedown, one seed, implies --envs 1
#     --detach       run in the background and return the prompt
#     --ladder       append the cumulative + solo ladder to the stage list
#     --contrib      append the three contribution stages (11, 12, 13)
#     --extra "..."  stage-specific train.py flags, passed after --
#     --dry-run      print the commands, run nothing

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROJ="${PROJ:-/workspace}"
ENV_DIR="${ENV_DIR:-$PROJ/envs/crafter}"

LADDER=(02 03 04 05 05b 06 06b 07 07b 08 08b 09 09b 10 10b 10c)
CONTRIB=(11 12 13)

STEPS=200000
ENVS=""
SEEDS="1,2"
NAME="live_run"
SMOKE=0
DETACH=0
DRY=0
EXTRA=""
STAGES=()

[ $# -eq 0 ] && { sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --steps)   STEPS="$2"; shift 2;;
    --envs)    ENVS="$2"; shift 2;;
    --seeds)   SEEDS="$2"; shift 2;;
    --name)    NAME="$2"; shift 2;;
    --extra)   EXTRA="$2"; shift 2;;
    --smoke)   SMOKE=1; shift;;
    --detach)  DETACH=1; shift;;
    --ladder)  STAGES+=("${LADDER[@]}"); shift;;
    --contrib) STAGES+=("${CONTRIB[@]}"); shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0;;
    -*) echo "ERROR: unknown option '$1'" >&2; exit 1;;
    *)  STAGES+=("$1"); shift;;
  esac
done

[ "${#STAGES[@]}" -gt 0 ] || { echo "ERROR: no stages given" >&2; exit 1; }

# One env per core, less one for the learner and one for the pod itself. The
# low-budget cards come with 8-16 vCPUs; above 8 envs the replay buffer, not
# the envs, becomes the limit, so capping there keeps RAM predictable.
if [ -z "$ENVS" ]; then
  CORES="$(nproc)"
  ENVS=$(( CORES - 2 ))
  [ "$ENVS" -lt 1 ] && ENVS=1
  [ "$ENVS" -gt 8 ] && ENVS=8
fi

if [ "$SMOKE" -eq 1 ]; then
  STEPS=2000
  SEEDS=1
  ENVS=1
  NAME="smoke"
fi

RUNLOG="$PROJ/logs/sweep-$(date +%Y%m%d-%H%M%S).out"

# --------------------------------------------------------------- the worker

# Everything the background process needs, as one script: a detached shell
# inherits no venv, which is the usual reason a nohup'd run dies in seconds
# with ModuleNotFoundError.
worker() {
  echo "=== sweep start $(date -Is)"
  echo "host     $(hostname)"
  echo "stages   ${STAGES[*]}"
  echo "steps    $STEPS   envs $ENVS   seeds $SEEDS   name $NAME"
  [ -n "$EXTRA" ] && echo "extra    $EXTRA"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/gpu      /'
  echo

  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: no CUDA. Refusing to burn paid hours on CPU training." >&2
    exit 1
  }

  # The envs are the parallel part; torch's intra-op threads would fight them
  # for the same cores.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export PYTHONUNBUFFERED=1

  local failed=()
  for s in "${STAGES[@]}"; do
    echo "=== stage $s  $(date -Is)"
    local args=(--steps "$STEPS" --envs "$ENVS" --seeds "$SEEDS" --name "$NAME"
                --logdir "${TMPDIR:-/tmp}/crafter-sweep-$$")
    [ "$SMOKE" -eq 1 ] && args+=(--smoke)
    if [ -n "$EXTRA" ]; then
      # Unquoted on purpose: it has to split into separate words.
      # shellcheck disable=SC2086
      ./run.sh "$s" "${args[@]}" -- $EXTRA || failed+=("$s")
    else
      ./run.sh "$s" "${args[@]}" || failed+=("$s")
    fi
    echo
  done

  echo "=== sweep done $(date -Is)"
  if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
  else
    echo "all ${#STAGES[@]} stages ok"
  fi
  # The pod bills by wall clock whether or not anything is running, so say so.
  echo "nothing is running now -- stop the pod if you are done."
}

# --------------------------------------------------------------- dispatch

if [ "$DRY" -eq 1 ]; then
  echo "would run, sequentially:"
  for s in "${STAGES[@]}"; do
    echo "  ./run.sh $s --steps $STEPS --envs $ENVS --seeds $SEEDS --name $NAME${EXTRA:+ -- $EXTRA}"
  done
  echo "log would be $RUNLOG"
  exit 0
fi

# shellcheck disable=SC1091
[ -f "$ENV_DIR/bin/activate" ] && source "$ENV_DIR/bin/activate"

mkdir -p "$(dirname "$RUNLOG")"

if [ "$DETACH" -eq 0 ]; then
  worker 2>&1 | tee "$RUNLOG"
  exit "${PIPESTATUS[0]}"
fi

# setsid as well as nohup: RunPod's web terminal kills the whole process group
# when the tab closes, which nohup alone does not protect against.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ARGS=(--steps "$STEPS" --envs "$ENVS" --seeds "$SEEDS" --name "$NAME")
[ "$SMOKE" -eq 1 ] && ARGS+=(--smoke)
[ -n "$EXTRA" ] && ARGS+=(--extra "$EXTRA")

setsid nohup bash "$SELF" "${ARGS[@]}" "${STAGES[@]}" > "$RUNLOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PROJ/logs/sweep.pid"

echo "detached as pid $PID"
echo "log    $RUNLOG"
echo "pid    $PROJ/logs/sweep.pid"
echo
echo "watch it:   tail -f $RUNLOG"
echo "stop it:    bash runpod/stop.sh"
