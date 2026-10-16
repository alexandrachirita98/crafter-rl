# Stage 02 smoke test — plain DQN, 2,000 steps

Run on **local WSL2, GTX 1650 4.3 GB, device `cuda`**. Free, no RunPod.
Reproduce with:

```bash
python -m common.networks                      # shapes, instant
python -m common.replay                        # buffer invariants, instant
python stage_02_dqn/train.py --total-steps 2000 \
    --learning-starts 500 --buffer-size 5000   # ~40 s
```

## What this run is and is not

**This is a wiring test, not a measurement.** Per
[implementation-guide.md §0.4](../../../docs/implementation-guide.md): *"Never
launch a pod to find out whether the code runs. Every stage gets
`--total-steps 2000` locally first."* The question is *does it crash, do the
shapes match, does the loop wire up* — and the answer is yes to all three.

**It cannot show learning, and it did not.** Two reasons, both structural:

| | |
| :-- | :-- |
| Gradient steps taken | 1,500 (steps 500–2,000) |
| Target-network refreshes | **1** (only the one at step 0) |
| Epsilon over the run | **1.00 → 0.992** |

Epsilon decays linearly over `--eps-decay-steps 250000`, so at step 2,000 it
has moved 0.8% of the way. **Every action in this run was uniform random.** The
network was trained on 1,500 batches but never once consulted for a decision.

So the score below is a random-policy score with extra steps. Reading it as
"DQN scores 1.21%" would be wrong twice over: the policy was not DQN's, and 12
episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Score | **1.21%** |
| Episodes | **12** |
| Mean episode length | 163.9 steps |
| Mean episode reward | 1.267 (max 3.1) |
| Mean achievements per episode | 2.17 |
| Throughput | 51 fps, 0.7 min for 2,000 steps |
| Stage 01 random, 1M steps | 1.51% over 5,959 episodes |

**Verdict: pass, on the checks this run can actually make.** The 1.21% versus
stage 01's 1.51% is not a regression — it is 12 episodes against 5,959. One
episode here is worth 8.3 percentage points on any achievement, so the entire
gap is a single lucky or unlucky episode.

### The 51 fps, and why it is not a concern

Stage 01 managed **517 fps** on CPU. This run did 51 fps, a 10× slowdown, and
the cause is that it now does a forward pass for every action and a
forward+backward for every step, on a 4 GB GTX 1650 that is also driving a
desktop. At this rate 1M steps would take 5.4 hours locally.

That is exactly the arithmetic in
[§0.4](../../../docs/implementation-guide.md) that sends real runs to RunPod,
so it is expected rather than alarming. What it does confirm: **CUDA was
actually used** (`device cuda` in the log), so the GPU path works before we pay
for a pod.

## Per-achievement success rate

Over 12 episodes, so each one is a multiple of 8.33%. Non-zero only.

| achievement | ours | episodes | stage 01 @ 1M |
| :-- | --: | --: | --: |
| `wake_up` | 100.00% | 12/12 | 93.45% |
| `collect_sapling` | 50.00% | 6/12 | 51.85% |
| `place_plant` | 41.67% | 5/12 | 46.60% |
| `collect_wood` | 16.67% | 2/12 | 23.81% |
| `collect_drink` | 8.33% | 1/12 | 9.60% |

The remaining 17 are 0%, including `place_table`, which stage 01 hit 4% of the
time — at 12 episodes we would expect 0.5 of them, so seeing none is normal.

**The agreement here is the real result of this run.** Five achievements, all
within sampling range of the 1M random baseline, is evidence that
`common/env.py` is constructing the same environment stage 01 measured and that
`crafter.Recorder` is writing `stats.jsonl` in the layout
`analysis/read_metrics.py` expects. That is what a smoke test is for.

## What was verified

| Check | Result |
| :-- | :-- |
| Runs 2,000 steps without crashing | pass |
| Conv stack flattens to exactly 1024 | pass |
| `QNetwork` emits `(batch, 17)` | pass |
| Buffer successor invariant holds across a wrap | pass |
| `stats.jsonl` written to the expected path | pass |
| Learning step fires after `--learning-starts` | pass |
| CUDA used when available | pass |
| **Learning demonstrated** | **no — out of scope, see above** |

The four silent-failure traps from
[§3](../../../docs/implementation-guide.md) — missing `permute`, missing
`/255`, mask from `done` instead of `info['discount']`, target net never or
always refreshed — cannot be caught by a 2,000-step score. They were checked
separately: the first two by `python -m common.networks`, the third by reading
`common/env.bootstrap_mask`, the fourth by the gradient-flow check that showed
loss falling 0.19 → 0.017 with finite gradients on all 10 parameter tensors.

## Two things to fix before the next run

**1. `--log-every` hid the diagnostics.** It defaults to 10,000, so a
2,000-step run printed no step lines at all — meaning **no loss or mean-Q trace
was captured**, which are the two numbers §3's "done when" criteria are written
in terms of. Nothing is wrong with the run; the instrument was switched off.
Next smoke test:

```bash
python stage_02_dqn/train.py --total-steps 2000 --learning-starts 500 \
    --buffer-size 5000 --log-every 200
```

**2. The conda env matters, and the guide has it wrong.**
[§0.5](../../../docs/implementation-guide.md) says base has
`torch 2.13.0+cpu with no CUDA`. True for base, but this run was in the
`crafter` env, which has **torch 2.6.0+cu124 and working CUDA**. Same command
in base silently runs 10× slower on CPU. Always:

```bash
conda activate crafter
```

## Carry forward

**The local ceiling is now measured: 51 fps.** 1M steps is 5.4 h locally
against ~0.8 h of pure env time on a pod with more vCPUs. Confirms the §0.4
rule — local is for 2,000-step wiring tests, and nothing that reaches a slide.

**Nothing here licenses a pod launch yet.** The next step is still local: a
2,000-step run with `--log-every 200` to see loss falling and mean-Q rising,
then the ☁️ 200k × 2 seeds dev run (~30 min, ~$0.18) that §3's "done when"
criteria are actually written for — mean return clearly above random's ≈2.1,
with `collect_wood`, `collect_drink` and `eat_cow` reliable.

## Files

| File | What |
| :-- | :-- |
| `summary_seed1.json` | every number above, machine-readable |
| `train_seed1.log` | full stdout of the run |
| `stats_seed1.jsonl` | raw per-episode records from `crafter.Recorder`, 12 lines |
| `selftests.txt` | `common.networks` and `common.replay` output |
