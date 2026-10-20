# Training the stages on RunPod

The FEP/UPB cluster path uses SLURM: `setup_cluster.sh` once, then `sbatch
jobs/train.sh` with a job array for the seeds. RunPod has no scheduler, so the
same three jobs are done by the scripts in this directory:

| On the cluster | On RunPod | What it does |
| :-- | :-- | :-- |
| `setup_cluster.sh` | `runpod/setup_runpod.sh` | build the env once, verify GPU + scorer |
| `sbatch jobs/train.sh` | `runpod/sweep.sh` | run stages back to back, detached |
| `squeue` / `sacct` | `runpod/status.sh` | what is running, what has scored |
| `scancel` | `runpod/stop.sh` | kill the run **and** free the GPU |
| shared `/export` filesystem | `runpod/sync.sh` | code up, results down |

`run.sh` itself is unchanged and does the actual work on both. Everything here
is a wrapper around it, so a RunPod result and a cluster result are the same
artifacts in the same layout and are directly comparable.

For the cluster, see [CLUSTER-UPB.md](../docs/CLUSTER-UPB.md). The older
[CLUSTER.md](../docs/CLUSTER.md) describes the same single-box idea by hand;
this doc replaces it with scripts.

---

## 0. Which pod to rent

**Do not pay for a fast GPU.** Crafter steps the environment in pure Python,
one core per env, and the network is tiny — the GPU idles most of the time.
From this repo's own logged runs, on a 16-core box:

| config | fps | 200k steps |
| :-- | --: | --: |
| stage 03, 1 env | 88 | ~38 min |
| stage 03, 16 envs | 92 | ~37 min |
| stage 10 (DrQ + resets), 1 env | 64 | ~52 min |

Sixteen parallel envs bought **4 fps**. The bottleneck is not parallelism you
can buy more of; it is the single-threaded replay/learner path. Treat ~60–90
fps as what any low-budget pod will give you.

So pick on price, not performance:

- **A4000 (16 GB)** or **A5000 (24 GB)** — the sweet spot. Anything more is
  wasted on a 54k–3.7M parameter net.
- **8–16 vCPU.** More cores help more than a better card, but see the table
  above before expecting much.
- **~20 GB volume.** A 100k-transition buffer is ~2.5 GB in RAM, not on disk;
  the repo plus env is a few GB.
- **At least 16 GB RAM**, more if you raise `--buffer-size`.

Budget arithmetic with the measured numbers, at ~0.25 $/h for an A4000:

| what | seed-runs | hours | cost |
| :-- | --: | --: | --: |
| one 200k seed | 1 | ~0.6 | ~0.16 $ |
| the 02→10c ladder, 200k × 2 seeds | 32 | ~22 | ~5.60 $ |
| the three C stages, 200k × 2 seeds | 6 | ~5.1 | ~1.30 $ |
| one 1M final run | 1 | ~3.2 | ~0.80 $ |

**The pod bills while idle.** Stop it the moment a sweep finishes — that, not
the card you picked, is what overruns a small budget.

---

## 1. Get the code onto the pod

Pick a directory on the **persistent volume**. RunPod wipes the container disk
on stop; only `/workspace` survives.

```sh
export PROJ=/workspace
cd $PROJ
git clone <your repo url> crafter-rl
cd crafter-rl
```

### The clone is not enough — read this one

`.gitignore` excludes `/demos/` and `/docs/`. A fresh clone therefore has **no
demo episodes**, and `stage_12_c3_demonstrations` replays them. Run it on a
clone as-is and it trains with an empty demo buffer, scores like its base, and
looks like the contribution did nothing.

Push them from your laptop instead:

```sh
# on your LAPTOP, in the repo
bash runpod/sync.sh push-demos <port> <host>
```

Or regenerate them on the pod with `python record_demos.py`. `setup_runpod.sh`
checks for them and tells you which case you are in.

If you have uncommitted work, skip the clone and push the whole tree:

```sh
# on your LAPTOP
bash runpod/sync.sh push <port> <host>
```

`<port>` and `<host>` come from RunPod's Connect panel. For
`ssh root@213.173.105.83 -p 40022` that is port `40022`, host `213.173.105.83`.
`push` deliberately excludes `logdir/`, `results/` and `demos/`: code goes up,
results come down, and nothing you push can overwrite pod output.

---

## 2. Build the environment — once

```sh
cd /workspace/crafter-rl
bash runpod/setup_runpod.sh
```

This differs from `setup_cluster.sh` in two ways that matter on a billed box:

- **venv with `--system-site-packages`, no conda.** RunPod's PyTorch images
  already ship a working CUDA torch. The script detects it and keeps it,
  instead of re-downloading ~3 GB of wheels you are paying for by the minute.
  Force a fresh one with `FORCE_TORCH=1 CUDA_TAG=cu118 bash runpod/setup_runpod.sh`.
- **It warns if the repo is outside `$PROJ`.** That is the mistake that loses a
  night's results to a pod restart.

Then it verifies, and these checks are the point of running it:

- `torch.cuda.is_available()` is `True` — if this is `False` you would train on
  CPU for hours and still be billed for the GPU
- the Crafter env builds and steps, returning `(64, 64, 3)`
- `python -m common.validate_scorer` reproduces the published baseline scores,
  so the metric is trustworthy
- `demos/` is present, or it tells you stage 12 will be hollow

Activate it in every new shell:

```sh
source /workspace/envs/crafter/bin/activate
```

A fresh `ssh` session inherits nothing. `sweep.sh` sources it itself, so a
detached run does not depend on you remembering — this is the usual cause of a
backgrounded run dying in seconds with `ModuleNotFoundError`.

---

## 3. Smoke test before anything long

```sh
bash runpod/sweep.sh --smoke 10c
```

2,000 steps, a couple of minutes, in the foreground. It must finish without a
traceback, and the log must say `device cuda`. **Note the fps it prints** —
that is what the cost table in section 0 should be recomputed with for your
pod.

---

## 4. Run the stages

One stage per `run.sh` call, stages one after another, in a detached process:

```sh
# the full ladder, 200k x 2 seeds -- the screening budget
bash runpod/sweep.sh --ladder --detach

# the three contribution stages
bash runpod/sweep.sh --contrib --detach

# named stages, your own settings
bash runpod/sweep.sh --steps 200000 --seeds 1,2 --detach 03 10c

# a 1M final run
bash runpod/sweep.sh --steps 1000000 --seeds 1 --name final --detach 03

# stage-specific train.py flags
bash runpod/sweep.sh --detach --extra "--lambda-aux 0.3" 11
```

| Option | Meaning | Default |
| :-- | :-- | :-- |
| `--steps N` | env steps per seed, total across envs | `200000` |
| `--envs N` | parallel envs | cores − 2, capped at 8 |
| `--seeds a,b` | seeds per stage, run one after another | `1,2` |
| `--name DIR` | results subdirectory | `live_run` |
| `--smoke` | 2k shakedown, one seed, one env | — |
| `--detach` | background it and return the prompt | — |
| `--ladder` | append `02 03 04 05 05b 06 06b 07 07b 08 08b 09 09b 10 10b 10c` | — |
| `--contrib` | append `11 12 13` | — |
| `--extra "..."` | flags passed through to `train.py` after `--` | — |
| `--dry-run` | print the commands, run nothing | — |

Three things it does that a hand-rolled `nohup` does not:

- **`setsid` as well as `nohup`.** RunPod's web terminal kills the whole
  process group when you close the tab; `nohup` alone does not survive that.
- **Refuses to start without CUDA.** Better to fail in the first second than to
  discover it six paid hours later.
- **`|| failed+=(...)` per stage.** One broken variant costs you that variant,
  not every stage after it. The log ends with a `FAILED:` line listing them.

It also sets `OMP_NUM_THREADS=1`, since torch's intra-op threads otherwise
fight the envs for the same cores.

**One sweep at a time.** Two concurrent runs on one pod halve each other's fps
and you pay twice the hours for the same work. The sequential loop is the whole
design, not a limitation.

A note on `--envs`: given the measured 88 → 92 fps for 1 → 16 envs, the default
is conservative on purpose. Verify with your own smoke-test fps before assuming
more envs buy throughput on your pod.

---

## 5. Watch it

```sh
bash runpod/status.sh                       # pod, GPU, running jobs, all scores
tail -f /workspace/logs/sweep-*.out         # the sweep's own log
tail -f stage_03_double/results/live_run/train_seed1.log   # one run's output
```

Reading it:

- **Low `nvidia-smi` utilisation is normal**, not a bug to chase. Crafter steps
  in Python on the CPU.
- **Load average above your core count** means more is running than the pod can
  feed. If you started two sweeps, that is why both crawl.
- **A log that stops growing on a live process** is output buffering;
  `sweep.sh` exports `PYTHONUNBUFFERED=1`, so this should not happen.

---

## 6. Stop it

```sh
bash runpod/stop.sh
```

`run.sh` pipes `train.py` through `tee`, so killing the wrapper does not
reliably take python with it, and a surviving python keeps the GPU allocated —
the next run then OOMs or crawls and looks like a bad GPU. `stop.sh` kills the
process group, sweeps up leftovers, and prints `nvidia-smi`'s compute-apps list
so you can see the GPU is genuinely free.

---

## 7. Bring the results home — before stopping the pod

```sh
# on your LAPTOP
bash runpod/sync.sh pull <port> <host>
```

This pulls `stage_*/results/` into `results-from-runpod/` — JSON, logs, text
only; checkpoints stay on the pod. Do it **before** the pod stops: anything
outside the volume is gone on stop and there is no second chance.

Then compare:

```sh
python compare_runs.py
```

For the assignment's plots, the `analysis/` scripts want the raw
`logdir/crafter_reward-<name>/<seed>/stats.jsonl` layout:

```sh
bash runpod/sync.sh pull-logdir <port> <host>
```

Then **stop the pod.**

---

## 8. When it goes wrong

**`torch.cuda.is_available()` is False**
No GPU attached, or a CPU torch got in. Check `nvidia-smi`, then
`FORCE_TORCH=1 CUDA_TAG=cu118 bash runpod/setup_runpod.sh`. `sweep.sh` refuses
to run in this state, so you cannot burn hours on it by accident.

**`ModuleNotFoundError` seconds after launch**
The shell never activated the venv. `sweep.sh` handles this itself; if you
launched `run.sh` by hand, `source /workspace/envs/crafter/bin/activate` first.

**Stage 12 scores exactly like its base**
Its demo buffer is empty. `demos/` is gitignored — see section 1.

**The run vanished when I closed the browser tab**
It was not started with `--detach`. Relaunch; nothing recovers it.

**Everything gone after a pod restart**
The repo was on the container disk, not the volume. Keep it under
`/workspace`.

**`No space left on device`**
`du -sh /workspace/*` and `df -h`. Usually a forgotten `--keep-logdir` run.
Never enable `save_video`: one 10,000-step episode buffers ~7.8 GB in RAM and
will OOM the pod as well as filling the disk.

**Slower than the cluster on a better GPU**
Expected. The bottleneck is the Python env loop, not the card.

---

## Checklist

- [ ] pod rented on price, not GPU speed; repo under `/workspace`
- [ ] `bash runpod/setup_runpod.sh` passed every check, including the scorer
- [ ] `demos/` present on the pod if you intend to run stage 12
- [ ] `bash runpod/sweep.sh --smoke 10c` finished, said `device cuda`, fps noted
- [ ] hours estimated against the budget with that fps
- [ ] long runs launched with `--detach`; exactly **one** sweep at a time
- [ ] `bash runpod/sync.sh pull` done **before** stopping the pod
- [ ] `bash runpod/stop.sh` shows no compute apps, then the pod is **stopped**
