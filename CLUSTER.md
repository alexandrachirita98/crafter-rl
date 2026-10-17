# Running this repo on the UPB grid cluster

Simple version: `setup.sh` builds the environment once, `run.sh` runs one
variant and saves its results, and a SLURM script wraps `run.sh` so the long
runs go in the queue.

Deeper background on SLURM, quotas and etiquette is in
[docs/running-on-the-cluster.md](docs/running-on-the-cluster.md). This file is
the short practical path.

---

## 0. What you are logging into

```sh
ssh -X -o ServerAliveInterval=100 dumitru.cercel@fep.grid.pub.ro
```

`fep` is the **front-end**. You may edit files and submit jobs there. You may
**not** train there — the cluster rules are explicit about it, and a long job on
`fep` gets killed.

Everything heavy runs on a compute node, reached either with `srun` (interactive)
or `sbatch` (queued).

The project directory with the 500 GB quota is **`/export/projects/grai`**. Home
directories are small, so code, the conda env and the logs all live there.

| Partition | GPUs | Use it for |
| :-- | :-- | :-- |
| `haswell` (default) | none | the random baseline (stage 01) |
| `ucsx` | 3× A100 | training — first choice |
| `xl` | 2× P100 | training — fallback when `ucsx` is full |
| `dgxa100` / `dgxh100` | 8× A100 / H100 | usually contended; no faster for us |

Crafter is bottlenecked by the Python environment loop, not the GPU, so **one
GPU per run** is always enough. Asking for more just blocks a colleague.

---

## 1. Get the code onto the cluster

```sh
export PROJ=/export/projects/grai
mkdir -p $PROJ/$USER/{code,logs,envs}
cd $PROJ/$USER/code
git clone <your repo url> crafter-rl
cd crafter-rl
```

If you have uncommitted work, push it from your laptop instead:

```sh
# on your LAPTOP
rsync -avz --exclude '.venv' --exclude 'logdir' --exclude '__pycache__' \
  ~/crafter-rl/ dumitru.cercel@fep.grid.pub.ro:/export/projects/grai/$USER/code/crafter-rl/
```

Code goes up, results come down. Never rsync `logdir/` upward.

---

## 2. Build the environment — once

Get an interactive job first. Compiling wheels is exactly what `fep` must not do.

```sh
srun --partition=ucsx --gres=gpu:1 --cpus-per-task=8 --mem=16G \
     --time=01:00:00 --pty bash

cd /export/projects/grai/$USER/code/crafter-rl
bash setup.sh
```

`setup.sh` creates a conda env at `$PROJ/$USER/envs/crafter`, installs CUDA
torch plus the handful of packages the stages need, and then **verifies**:

- `torch.cuda.is_available()` is `True` — if this prints `False` you would be
  training on CPU for twelve hours and not know it
- the Crafter env builds and steps, returning a `(64, 64, 3)` observation
- `python -m common.validate_scorer` reproduces all seven published baseline
  scores to within 0.1 points, so our metric is trustworthy

It is safe to re-run; it reuses an existing env.

If the CUDA module name on your node differs, pass it in:

```sh
CUDA_TAG=cu118 bash setup.sh      # older driver
PY_VER=3.10 bash setup.sh         # different python
```

**Note the measured fps** from the smoke test in the next step — you need it to
set job walltimes.

---

## 3. One smoke test before anything long

Still in the interactive job:

```sh
./run.sh 10c --smoke
```

2,000 steps, about two minutes. It must finish without a traceback. Check that
the log says `device cuda` and not `device cpu`.

Then release the GPU — an idle interactive job holds it until you exit:

```sh
exit
```

---

## 4. How `run.sh` works

One variant per call. It saves the same files the committed `smoke_test`
directories contain, under `results/live_run/` by default.

```sh
./run.sh <stage> [options] [-- stage-specific flags]
```

| Option | Meaning | Default |
| :-- | :-- | :-- |
| `--steps N` | environment steps | `200000` |
| `--seed N` | one seed | `1` |
| `--seeds a,b` | several seeds, one after another | — |
| `--name DIR` | results subdirectory | `live_run` |
| `--smoke` | 2,000 steps with a small buffer, for shakedowns | — |
| `--logdir DIR` | where `train.py` writes raw output | scratch, auto-deleted |
| `--keep-logdir` | keep that raw output | — |
| `--dry-run` | print the command, run nothing | — |
| `--list` | list the stage names | — |

The stage can be the short key or the full name — `10c`, `stage_10c`, and
`stage_10c_no_per` all work. `./run.sh --list` prints them all.

What lands in `<stage>/results/<name>/`:

```
train_seed1.log        full stdout, including the score at the end
stats_seed1.jsonl      one JSON object per episode -- this is what gets scored
summary_seed1.json     score, fps, episodes, per-achievement success rates
selftests.txt          the stage's own --test-* check, for stages that have one
command_seed1.txt      the exact command, pasteable
```

These are the same filenames the committed `smoke_test/` directories use, so a
cluster run and a local smoke test are directly comparable.

Stages 02-10 share one argument surface (they inherit it from stage 02), so the
same options work on all of them. Stages with a `--test-*` self-check run it
first and save the output; stages 02 and 03 have none, so they produce no
`selftests.txt`.

Examples:

```sh
./run.sh 06 --steps 200000 --seed 1          # one seed of QR-DQN
./run.sh 10 --seeds 1,2 --steps 200000       # both screening seeds
./run.sh 08 --steps 1000000 --name final     # a 1M run into results/final/
./run.sh 09 -- --ez-mu 4.0                   # pass a stage-specific flag
./run.sh 02 --dry-run                        # see the command first
```

Anything after `--` goes straight to `train.py`, which is how you reach the
per-stage flags (`--n-step`, `--n-quantiles`, `--per-alpha`, `--ez-mu`,
`--reset-every`, …).

---

## 5. The job script

`run.sh` is what the SLURM job calls. Create `jobs/train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=crafter
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=1-2

set -euo pipefail
PROJ=/export/projects/grai

module load libraries/cuda-13.0 2>/dev/null || true
# conda activate needs this line first in a non-interactive shell. Forgetting
# it is the single most common batch-script bug.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $PROJ/$USER/envs/crafter

cd $PROJ/$USER/code/crafter-rl

echo "node=$(hostname) job=$SLURM_JOB_ID seed=$SLURM_ARRAY_TASK_ID"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"

./run.sh "${STAGE:?set STAGE}" \
  --steps "${STEPS:-200000}" \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --name "${NAME:-live_run}"
```

The `assert torch.cuda.is_available()` fails the job in ten seconds instead of
letting it crawl on CPU for hours.

`--array=1-2` gives you seeds 1 and 2 as two independent jobs, which is the
screening convention in [docs/ablation-plan.md](docs/ablation-plan.md).

Submit one stage:

```sh
mkdir -p logs
STAGE=10c STEPS=200000 sbatch jobs/train.sh
squeue --me
```

---

## 6. Running every variant, 02 → 10

The screening budget is **200k steps × 2 seeds** per variant. That is what the
ablation plan calls for; 1M is reserved for the final few.

There are three families:

- **02–10** — the cumulative stack. Each one adds a component on top of the
  previous, so stage 10 is everything at once.
- **05b–10b** — the *solo* variants. Each adds its one component to the plain
  stage-03 base instead of stacking, which is what isolates its individual
  contribution.
- **10c** — stage 10 with PER removed, to test whether PER earns its place.

### Submit them all

```sh
cd $PROJ/$USER/code/crafter-rl
mkdir -p logs

# the cumulative stack
for s in 02 03 04 05 06 07 08 09 10; do
  STAGE=$s STEPS=200000 sbatch --job-name=crafter-$s jobs/train.sh
done

# the solo ablations
for s in 05b 06b 07b 08b 09b 10b; do
  STAGE=$s STEPS=200000 sbatch --job-name=crafter-$s jobs/train.sh
done

# PER removed
STAGE=10c STEPS=200000 sbatch --job-name=crafter-10c jobs/train.sh
```

That is 16 stages × 2 seeds = **32 jobs**. Submit the cumulative stack first and
let it finish before the solos if the queue is busy — the stack is what the
headline numbers come from.

### Or run them one at a time interactively

Useful while you are still debugging. One at a time, never in parallel on a
single node — parallel runs contend for CPU and stall.

```sh
for s in 02 03 04 05 06 07 08 09 10 05b 06b 07b 08b 09b 10b 10c; do
  ./run.sh $s --steps 200000 --seeds 1,2
done
```

### Stage 01, the random baseline

No neural network, so no GPU. Do not waste an allocation on it.

```sh
sbatch --partition=haswell --cpus-per-task=1 --mem=2G --time=03:00:00 \
       --wrap="source \$(conda info --base)/etc/profile.d/conda.sh; \
               conda activate $PROJ/\$USER/envs/crafter; \
               cd $PROJ/\$USER/code/crafter-rl; \
               ./run.sh 01 --steps 1000000 --seed 1"
```

### Walltime

Measured locally on CPU the stages run 16–28 fps; on a GPU node expect several
times that. Work it out from your own measured rate rather than guessing:

> at *r* fps, N steps takes `N / r / 3600` hours

Add about 30 % headroom. `--time=04:00:00` is a sane starting point for 200k on
a GPU node, `--time=12:00:00` for 1M. SLURM kills a job the moment the walltime
expires, with no grace period and no traceback — the most common way a run dies
at 90 %.

---

## 7. Watch the jobs

```sh
squeue --me                           # what is queued or running
tail -f logs/crafter-10c-*.out        # live output
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scontrol show job <jobid>             # why is it still pending?
scancel <jobid>                       # cancel one
scancel -u $USER                      # cancel everything of yours
```

`(Resources)` next to a pending job means the partition is full — wait, or
switch to `xl`.

---

## 8. Bring the results home

The files that matter are small — JSON and text only. Leave checkpoints on the
cluster.

```sh
# on your LAPTOP
rsync -avz \
  dumitru.cercel@fep.grid.pub.ro:/export/projects/grai/$USER/code/crafter-rl/'stage_*/results/' \
  ~/crafter-rl/results-from-cluster/
```

Each `summary_seed*.json` already carries the score, so comparing variants is
just reading them:

```sh
cd ~/crafter-rl
python - <<'EOF'
import glob, json
rows = []
for f in sorted(glob.glob('stage_*/results/*/summary_seed*.json')):
    d = json.load(open(f))
    rows.append((d['stage'], d['seed'], d['score_percent'], d['episodes']))
print(f"{'stage':<30} {'seed':>4} {'score':>8} {'episodes':>9}")
for stage, seed, score, eps in rows:
    print(f'{stage:<30} {seed:>4} {score:>7.2f}% {eps:>9}')
EOF
```

For the plots required by the assignment, copy the `stats.jsonl` files into the
`logdir/crafter_reward-<name>/<seed>/` layout the analysis scripts expect, then:

```sh
cd analysis
python read_metrics.py      # note: its read_stats() calls are hardcoded to the
                            # paper's baselines -- add a line for your own stage
python print_scores.py
python plot_reward.py
python plot_spectrum.py
```

---

## 9. When it goes wrong

**`torch.cuda.is_available()` is False in a GPU job**
Either `--gres=gpu:1` is missing (the default is zero GPUs) or a CPU torch wheel
got installed. Check `echo $CUDA_VISIBLE_DEVICES` and `nvidia-smi`, then re-run
`setup.sh` with the right `CUDA_TAG`.

**`ModuleNotFoundError` in a batch job but not interactively**
The script skipped `source "$(conda info --base)/etc/profile.d/conda.sh"` before
`conda activate`.

**`Invalid partition name`**
You do not have access to it. `sacctmgr show user $USER withassoc` lists what
you do.

**Job dies exactly at the walltime with no traceback**
SLURM killed it. Raise `--time`.

**`Disk quota exceeded`**
`du -sh /export/projects/grai/$USER/*`. Usually checkpoints. Never enable
`save_video` — one 10,000-step episode buffers about 7.8 GB in RAM and will OOM
the node as well as filling the quota.

**Permission denied writing to `/export/projects/grai`**
Your ACL was not set, or only the non-default half was. **Both** commands are
needed, per the [ACL guide](https://guides.upb.ro/docs/grid/acls):

```sh
setfacl -m -R u:your.name:rwx /export/projects/grai/
setfacl -d -m u:your.name:rwx /export/projects/grai/
```

**Training slower than expected on an A100**
Normal. Crafter steps the environment in pure Python on one core, so the GPU
idles most of the time. More `--cpus-per-task` helps more than a bigger GPU.

---

## Checklist

- [ ] `ssh` to `fep` works
- [ ] code at `/export/projects/grai/$USER/code/crafter-rl`
- [ ] `bash setup.sh` run **in an interactive job**, all its checks passed
- [ ] `torch.cuda.is_available()` printed `True`
- [ ] `./run.sh 10c --smoke` finished, log said `device cuda`
- [ ] fps measured, `--time` set from it with headroom
- [ ] `jobs/train.sh` activates conda via `conda.sh` and asserts CUDA
- [ ] stages 02–10 and the solos submitted, 2 seeds each
- [ ] stage 01 on `haswell`, no GPU
- [ ] `results/` rsynced back; scores compared
