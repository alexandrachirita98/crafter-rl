#!/bin/bash
# Run one stage and collect its artifacts into <stage>/results/live_run/,
# in the same layout as the committed smoke_test dirs:
#
#     train_seed<N>.log      full stdout of the run
#     stats_seed<N>.jsonl    per-episode stats from the run's logdir
#     summary_seed<N>.json   score, fps, per-achievement rates
#     selftests.txt          the stage's --test-* self-check, if it has one
#     command_seed<N>.txt    the exact command line, for reproducibility
#
# Usage:
#     ./run.sh <stage> [options] [-- extra train.py flags]
#
#     ./run.sh 10c                                   # defaults: 200k, seed 1
#     ./run.sh stage_06_qrdqn --steps 200000 --seed 2
#     ./run.sh 10 --seeds 1,2 --steps 200000         # both seeds, sequentially
#     ./run.sh 02 --smoke                            # 2k-step shakedown
#     ./run.sh 08 --steps 1000000 --name final       # -> results/final/
#     ./run.sh 09 -- --ez-mu 4.0                     # stage-specific flag
#     ./run.sh 03 --envs 16 --steps 200000 --name 200k_v16   # parallel envs
#
# Options:
#     --steps N      env steps, TOTAL across envs   (default 200000)
#     --envs N       parallel envs          (default 1, reproduces 02-10)
#     --seed N       single seed            (default 1)
#     --seeds a,b    several seeds, run one after another
#     --name DIR     results subdir name    (default live_run)
#     --smoke        2000 steps, warmup 500, buffer 5000, log-every 200
#     --logdir DIR   where train.py writes  (default $TMPDIR or ./logdir)
#     --keep-logdir  do not delete the raw logdir after copying artifacts
#     --dry-run      print what would run, run nothing
#     --list         list the available stages and exit

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# ------------------------------------------------------------------ stages

# Every stage dir that has a train.py, in order.
mapfile -t STAGES < <(find . -maxdepth 1 -name 'stage_*' -type d \
  -exec test -f '{}/train.py' ';' -print | sed 's|^\./||' | sort)

list_stages() {
  echo "available stages:"
  for s in "${STAGES[@]}"; do
    # The short key is what you can pass instead of the full name.
    echo "  ${s#stage_}" | sed 's/_.*//' | tr -d '\n'
    echo "   $s"
  done
}

# Resolve "10c", "stage_10c", "stage_10c_no_per" -> stage_10c_no_per.
resolve_stage() {
  local want="$1" s
  for s in "${STAGES[@]}"; do [ "$s" = "$want" ] && { echo "$s"; return; }; done
  for s in "${STAGES[@]}"; do [ "$s" = "stage_$want" ] && { echo "$s"; return; }; done
  # Prefix match on the numeric part: 10c -> stage_10c_*
  local hits=()
  for s in "${STAGES[@]}"; do
    case "$s" in stage_"$want"_*|stage_"$want") hits+=("$s");; esac
  done
  if [ "${#hits[@]}" -eq 1 ]; then echo "${hits[0]}"; return; fi
  if [ "${#hits[@]}" -gt 1 ]; then
    echo "ERROR: '$want' is ambiguous: ${hits[*]}" >&2; return 1
  fi
  echo "ERROR: no stage matching '$want'. Try ./run.sh --list" >&2
  return 1
}

# ------------------------------------------------------------------ args

STEPS=200000
ENVS=1
SEEDS=""
SEED=1
NAME="live_run"
SMOKE=0
LOGROOT=""
KEEP_LOGDIR=0
DRY=0
EXTRA=()

[ $# -eq 0 ] && { sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 1; }

case "${1:-}" in
  --list) list_stages; exit 0;;
  -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0;;
esac

STAGE="$(resolve_stage "$1")"; shift

while [ $# -gt 0 ]; do
  case "$1" in
    --steps)       STEPS="$2"; shift 2;;
    --envs)        ENVS="$2"; shift 2;;
    --seed)        SEED="$2"; shift 2;;
    --seeds)       SEEDS="$2"; shift 2;;
    --name)        NAME="$2"; shift 2;;
    --logdir)      LOGROOT="$2"; shift 2;;
    --smoke)       SMOKE=1; shift;;
    --keep-logdir) KEEP_LOGDIR=1; shift;;
    --dry-run)     DRY=1; shift;;
    --)            shift; EXTRA=("$@"); break;;
    *) echo "ERROR: unknown option '$1' (stage-specific flags go after --)" >&2
       exit 1;;
  esac
done

[ -n "$SEEDS" ] || SEEDS="$SEED"

# Smoke defaults. Ask the stage what it accepts rather than guessing: the DQN
# stages inherit their args from stage 02, but stage 01 is a random policy with
# no buffer or warmup at all.
SMOKE_ARGS=()
if [ "$SMOKE" -eq 1 ]; then
  STEPS=2000
  HELP="$(python "$STAGE/train.py" --help 2>/dev/null || true)"
  for f in --warmup-steps:500 --buffer-size:5000 --log-every:200; do
    flag="${f%%:*}"; val="${f##*:}"
    if printf '%s' "$HELP" | grep -q -- "$flag"; then
      SMOKE_ARGS+=("$flag" "$val")
    fi
  done
fi

# train.py writes <logdir>/crafter_reward-<stage>/<seed>/stats.jsonl. Keep that
# off the repo tree by default so a run never dirties git.
if [ -z "$LOGROOT" ]; then
  LOGROOT="${TMPDIR:-/tmp}/crafter-run-$$"
fi

RESULTS="$REPO/$STAGE/results/$NAME"

echo "stage    $STAGE"
echo "steps    $STEPS"
echo "envs     $ENVS"
echo "seeds    $SEEDS"
echo "results  $STAGE/results/$NAME"
echo "logdir   $LOGROOT"
[ "${#EXTRA[@]}" -gt 0 ] && echo "extra    ${EXTRA[*]}" || true
echo

# ------------------------------------------------------------------ selftest

[ "$DRY" -eq 1 ] || mkdir -p "$RESULTS"

TEST_FLAG="$({ grep -oE "'--test-[a-z-]+'" "$STAGE/train.py" || true; } \
  | head -1 | tr -d "'")"
if [ -n "$TEST_FLAG" ]; then
  echo "--- self-test: $STAGE/train.py $TEST_FLAG"
  if [ "$DRY" -eq 1 ]; then
    echo "DRY: python $STAGE/train.py $TEST_FLAG"
  else
    {
      echo "\$ python $STAGE/train.py $TEST_FLAG"
      python "$STAGE/train.py" "$TEST_FLAG" 2>&1
    } | tee "$RESULTS/selftests.txt"
    echo
  fi
fi

# ------------------------------------------------------------------ run

for s in ${SEEDS//,/ }; do
  LOG="$RESULTS/train_seed$s.log"
  RUNDIR="$LOGROOT/crafter_reward-$STAGE/$s"
  STATS_SRC="$RUNDIR/stats.jsonl"

  CMD=(python "$STAGE/train.py"
       --total-steps "$STEPS"
       --num-envs "$ENVS"
       --seed "$s"
       --logdir "$LOGROOT")
  [ "${#SMOKE_ARGS[@]}" -gt 0 ] && CMD+=("${SMOKE_ARGS[@]}") || true
  [ "${#EXTRA[@]}" -gt 0 ] && CMD+=("${EXTRA[@]}") || true

  echo "--- seed $s: ${CMD[*]}"
  if [ "$DRY" -eq 1 ]; then
    echo "DRY: would log to $LOG"
    continue
  fi

  # Record the command with a stable --logdir, not this run's scratch path,
  # so the line can be pasted back verbatim.
  printf '%s\n' "${CMD[*]//$LOGROOT/logdir}" > "$RESULTS/command_seed$s.txt"

  # tee so a long run is watchable live and still captured in full.
  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: seed $s exited $rc. Log kept at $LOG" >&2
    exit "$rc"
  fi

  # Copy the stats the Recorder wrote into the results dir. With --envs > 1
  # there is one stats.jsonl per shard; concatenate them so the artifact
  # layout stays stats_seed<N>.jsonl either way. Concatenating is correct for
  # scoring: the score is computed over the POOLED episodes, and every line is
  # one finished episode, so order does not matter. The per-shard files are
  # kept alongside for anyone who wants per-env curves.
  mapfile -t SHARDS < <(ls "$RUNDIR"/shard*/stats.jsonl 2>/dev/null || true)
  if [ "${#SHARDS[@]}" -gt 0 ]; then
    cat "${SHARDS[@]}" > "$RESULTS/stats_seed$s.jsonl"
    echo "wrote $STAGE/results/$NAME/stats_seed$s.jsonl"
    echo "  pooled from ${#SHARDS[@]} shards"
    mkdir -p "$RESULTS/shards_seed$s"
    for sh in "${SHARDS[@]}"; do
      n="$(basename "$(dirname "$sh")")"
      cp "$sh" "$RESULTS/shards_seed$s/$n.jsonl"
    done
  elif [ -f "$STATS_SRC" ]; then
    cp "$STATS_SRC" "$RESULTS/stats_seed$s.jsonl"
    echo "wrote $STAGE/results/$NAME/stats_seed$s.jsonl"
  else
    echo "WARNING: no stats.jsonl under $RUNDIR" >&2
  fi

  # Mechanical summary: score, fps, per-achievement rates.
  if [ -f "$RESULTS/stats_seed$s.jsonl" ]; then
    STAGE="$STAGE" SEED="$s" STEPS="$STEPS" NAME="$NAME" ENVS="$ENVS" \
    LOG="$LOG" STATS="$RESULTS/stats_seed$s.jsonl" \
    OUT="$RESULTS/summary_seed$s.json" \
    python - <<'PY'
import json, os, pathlib, re, sys
sys.path.append(os.environ['PWD'])
from common.metrics import ACHIEVEMENTS, score_run

stats = pathlib.Path(os.environ['STATS'])
log = pathlib.Path(os.environ['LOG']).read_text()
score, percents, n = score_run(stats)

eps = [json.loads(l) for l in stats.read_text().splitlines() if l.strip()]
lengths = [e['length'] for e in eps]
rewards = [e['reward'] for e in eps]

# "done: 11 episodes in 1.6 min (21 fps)" -- written by train.py.
m = re.search(r'done: (\d+) episodes in ([\d.]+) min \((\d+) fps\)', log)
wall, fps = (float(m.group(2)), int(m.group(3))) if m else (None, None)
device = (re.search(r'device (\w+)', log) or [None, None])[1]

# The eps the score was actually measured at. A score at eps=0.32 is a third
# uniform-random actions and is not comparable to one at eps=0.05; the first
# 200k ladder was read as if it were. Taken from the last progress line.
eps_seen = re.findall(r'eps=([\d.]+)', log)
final_eps = float(eps_seen[-1]) if eps_seen else None

# Both default to 1M-sized values, so on a 200k run they are part of the
# result, not background. Read off the command rather than re-plumbed.
cmd = pathlib.Path(os.environ['OUT']).parent.joinpath(
    f"command_seed{os.environ['SEED']}.txt").read_text().strip()
def flag(name, default):
    m = re.search(rf'--{name}\s+(\S+)', cmd)
    return int(m.group(1)) if m else default

summary = {
    'stage': os.environ['STAGE'],
    'run_kind': os.environ['NAME'],
    'command': cmd,
    'env_steps': int(os.environ['STEPS']),
    'num_envs': int(os.environ.get('ENVS', 1)),
    'seed': int(os.environ['SEED']),
    'device': device,
    'final_epsilon': final_eps,
    'warmup_steps': flag('warmup-steps', 20_000),
    'buffer_size': flag('buffer-size', 100_000),
    'episodes': n,
    'wall_minutes': wall,
    'fps': fps,
    'score_percent': round(score, 3),
    'mean_episode_length': round(sum(lengths) / len(lengths), 1) if eps else None,
    'mean_episode_reward': round(sum(rewards) / len(rewards), 4) if eps else None,
    'max_episode_reward': max(rewards) if eps else None,
    'success_percent': {k: round(float(v), 4)
                        for k, v in zip(ACHIEVEMENTS, percents)},
}
pathlib.Path(os.environ['OUT']).write_text(json.dumps(summary, indent=2) + '\n')
print(f"score {score:.2f}%  over {n} episodes  ({fps} fps)")
print('wrote', os.environ['OUT'])
PY
  fi
  echo
done

# ------------------------------------------------------------------ cleanup

if [ "$DRY" -eq 0 ] && [ "$KEEP_LOGDIR" -eq 0 ] \
   && [ -d "$LOGROOT" ] && [[ "$LOGROOT" == *crafter-run-* ]]; then
  rm -rf "$LOGROOT"
fi

if [ "$DRY" -eq 0 ]; then
  echo "=== artifacts in $STAGE/results/$NAME ==="
  ls -1 "$RESULTS"
fi
