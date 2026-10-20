#!/bin/bash
# One screen of "what is this pod doing and what has it produced".
#
#     bash runpod/status.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PROJ="${PROJ:-/workspace}"

echo "=== pod ==="
echo "host    $(hostname)"
echo "uptime  $(uptime | sed 's/.*up //')"
nproc | sed 's/^/cores   /'
free -g | awk '/^Mem:/ {print "ram     " $3 " / " $2 " GB used"}'
df -h "$PROJ" 2>/dev/null | tail -1 | awk '{print "volume  " $4 " free of " $2}'

echo
echo "=== gpu ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
# Low GPU utilisation is expected: Crafter steps in Python on the CPU.

echo
echo "=== running ==="
pgrep -af 'train.py|sweep.sh' || echo "nothing running"

echo
echo "=== latest sweep log ==="
LATEST="$(ls -t "$PROJ"/logs/sweep-*.out 2>/dev/null | head -1 || true)"
if [ -n "$LATEST" ]; then
  echo "$LATEST"
  tail -5 "$LATEST"
else
  echo "none in $PROJ/logs"
fi

echo
echo "=== scores so far ==="
python - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob('stage_*/results/*/summary_seed*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append((d.get('stage', '?'), d.get('run_kind', '?'), d.get('seed'),
                 d.get('score_percent'), d.get('episodes'), d.get('fps')))
if not rows:
    print('no summary_seed*.json yet')
else:
    print(f"{'stage':<32} {'run':<12} {'seed':>4} {'score':>7} {'eps':>6} {'fps':>5}")
    for stage, kind, seed, score, eps, fps in rows:
        sc = f'{score:.2f}%' if isinstance(score, (int, float)) else '-'
        print(f'{stage:<32} {kind:<12} {str(seed):>4} {sc:>7} {str(eps):>6} {str(fps):>5}')
PY
