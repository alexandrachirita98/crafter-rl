#!/bin/bash
# Move code and results between your laptop and a RunPod pod.
#
# Run this FROM YOUR LAPTOP, not on the pod.
#
#     bash runpod/sync.sh push        <port> <host>   # code up (no results, no logdir)
#     bash runpod/sync.sh push-demos  <port> <host>   # demos/ up -- stage 12 needs it
#     bash runpod/sync.sh pull        <port> <host>   # results down
#     bash runpod/sync.sh pull-logdir <port> <host>   # raw stats.jsonl down, for the plots
#
# <port> and <host> are what RunPod's "Connect" panel shows, e.g.
#     ssh root@213.173.105.83 -p 40022 -i ~/.ssh/id_ed25519
# gives host 213.173.105.83 and port 40022.
#
# Why push-demos is separate: demos/ is in .gitignore, so a git clone on the
# pod gets no demos at all, and stage 12 would then train without them and
# look like the contribution failed. Same for docs/.
#
# Env vars:
#   REMOTE   repo path on the pod   (default /workspace/crafter-rl)
#   SSHKEY   identity file          (default: whatever ssh picks)
#   USER_AT  remote user            (default root)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-/workspace/crafter-rl}"
USER_AT="${USER_AT:-root}"

MODE="${1:-}"
PORT="${2:-}"
HOST="${3:-}"

[ -n "$MODE" ] && [ -n "$PORT" ] && [ -n "$HOST" ] || {
  sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 1; }

SSH="ssh -p $PORT"
[ -n "${SSHKEY:-}" ] && SSH="$SSH -i $SSHKEY"
TARGET="$USER_AT@$HOST"

case "$MODE" in

  push)
    # Code only. logdir/ and results/ are deliberately excluded: logdir is
    # large and regenerated, and pushing results would overwrite pod output
    # with your older local copies.
    rsync -avz --progress -e "$SSH" \
      --exclude '.git' --exclude '.venv' --exclude 'venv' \
      --exclude '__pycache__' --exclude '*.pyc' --exclude '*.egg-info' \
      --exclude 'logdir/' --exclude 'logs/' \
      --exclude 'results/' --exclude 'demos*/' \
      --exclude 'archive/' --exclude '*.npz' --exclude '*.pt' \
      "$REPO/" "$TARGET:$REMOTE/"
    echo
    echo "code up. demos/ was NOT included -- push it separately if you need stage 12:"
    echo "  bash runpod/sync.sh push-demos $PORT $HOST"
    ;;

  push-demos)
    [ -n "$(ls "$REPO"/demos/*.npz 2>/dev/null)" ] || {
      echo "ERROR: no $REPO/demos/*.npz to push. Record them first:"
      echo "  python record_demos.py"; exit 1; }
    echo "pushing $(ls "$REPO"/demos/*.npz | wc -l) demo episodes"
    rsync -avz --progress -e "$SSH" \
      "$REPO/demos/" "$TARGET:$REMOTE/demos/"
    ;;

  pull)
    # The small files only: JSON, logs, text. Checkpoints stay on the pod.
    mkdir -p "$REPO/results-from-runpod"
    rsync -avz --progress -e "$SSH" \
      --include 'stage_*/' --include 'stage_*/results/***' \
      --exclude '*' \
      --exclude '*.pt' --exclude '*.npz' \
      "$TARGET:$REMOTE/" "$REPO/results-from-runpod/"
    echo
    echo "pulled into results-from-runpod/. Compare them with:"
    echo "  python compare_runs.py"
    ;;

  pull-logdir)
    # Only needed for the analysis/ plot scripts, which read the
    # logdir/crafter_reward-<name>/<seed>/stats.jsonl layout.
    mkdir -p "$REPO/logdir-from-runpod"
    rsync -avz --progress -e "$SSH" \
      --include '*/' --include 'stats.jsonl' --exclude '*' \
      "$TARGET:$REMOTE/logdir/" "$REPO/logdir-from-runpod/"
    ;;

  *) echo "ERROR: unknown mode '$MODE'" >&2
     sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 1;;
esac
