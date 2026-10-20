#!/bin/bash
# Stop a detached sweep and make sure the GPU is actually free again.
#
#     bash runpod/stop.sh
#
# run.sh pipes train.py through tee, so killing the wrapper does not reliably
# take python with it. A leftover python keeps the GPU allocated and the next
# run then either OOMs or crawls, which on a billed box looks like a bad GPU.

set -uo pipefail

PROJ="${PROJ:-/workspace}"
PIDFILE="$PROJ/logs/sweep.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "killing sweep process group $PID"
    kill -- "-$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
    sleep 2
  else
    echo "pid $PID in $PIDFILE is not running"
  fi
  rm -f "$PIDFILE"
else
  echo "no $PIDFILE"
fi

# Whatever survived the group kill.
if pgrep -f 'stage_.*/train.py' >/dev/null 2>&1; then
  echo "leftover train.py processes:"
  pgrep -af 'stage_.*/train.py'
  pkill -f 'stage_.*/train.py' || true
  sleep 3
  pkill -9 -f 'stage_.*/train.py' 2>/dev/null || true
fi

echo
echo "--- still running? ---"
pgrep -af 'train.py|run.sh|sweep.sh' || echo "nothing"

echo
echo "--- GPU ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
  || echo "no nvidia-smi"
echo
echo "If no process is listed above, the pod is idle -- stop it to stop paying."
