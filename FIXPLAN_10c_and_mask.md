# Stage 10c multi-env crash: fix, verification, and the cluster run order

Written 2026-10-21, after the pull that brought `fd3b5e9` ("results after
marking") and `eeea629` ("marked 10c").

**Status: the code changes are implemented and verified locally.** What remains
is GPU time on the cluster. Section 3 is the ordered command list.

---

## 1. What was wrong

Both seeds of the new stage 10c run died ~20k steps in, identically, in both the
`fd3b5e9` launch and the `eeea629` re-run:

```
IndexError: index 1 is out of bounds for axis 0 with size 1
  stage_02_dqn/train.py:347 in train
    agent.learn(i, states[i], actions[i], rewards[i], states_[i], ...)
```

`stage_10c_no_per/train.py`'s `step()` override was written for a single
environment: every index was a literal `[0]`, and two of its three return paths
built a length-1 array. The train loop then iterated `for i in range(num_envs)`
over 16 envs and indexed `actions[1]`.

The `loss=nan, meanQ=nan` on the lines before the traceback is just the warmup
window, not a second bug.

**Why it only surfaced now.** The previously-working 10c run
(`results/200k_eps_fix`, 2.001 / 2.125) passed no `--num-envs`, and the parser
defaults to 1, so every `[0]` was correct. The new run added `--num-envs 16`.
This is a latent bug the env-count switch exposed, not a regression from the
masking work.

**Why the self-test missed it.** It drove the agent with `agent.step(s[None])` —
a batch of exactly one — and built the agent without `num_envs`. It reproduced
the only case that worked.

**The more important half of the bug.** Even without the `IndexError`, the old
code kept *one* walk shared across all 16 envs, so no env ever held an action for
its own k steps — ez-greedy itself was destroyed, not merely mis-indexed. So no
multi-env 10c number recorded before this fix would have been meaningful.

**Scope.** Only two files hardcode `[0]` in `step()`: `stage_10c_no_per`
(the bug) and `stage_09b_ez_greedy_solo` (a deliberate single-env diagnostic —
correct as-is). `stage_10_drq_resets` inherits stage 09's correct loop.

---

## 2. The code changes (done)

### A1 — per-env loop in 10c's `step()`

`stage_10c_no_per/train.py`. Replaced the single-env body with stage 09's
per-env loop (`stage_09_ez_greedy/train.py:50-87`), which was already correct.
10c's own docstring says the overrides are "spelled out ... each differing only
in which class `super()` reaches", so this restores the original intent. The
algorithm is unchanged.

Now: `actions` is always length `n`; each env keeps its own walk; and the
shared `torch.rand(1)` coin flip became a per-env `np.random.rand()` draw,
matching stage 09. One batched forward pass still serves all envs wanting a
greedy action.

### A2 — self-test drives a real batch

Same file, `test_no_per`. Three changes:

- new `n_envs=4` parameter, passed as `num_envs=n_envs` to the agent (without
  this, `_repeat_left` stays length 1 and the new loop would fail for the wrong
  reason);
- the driver builds `(n_envs, 64, 64, 3)` observations, calls `agent.learn` once
  per env, and asserts `np.shape(actions) == (n_envs,)` — reported as a new
  `step() gave one action per env` line folded into the PASS/FAIL verdict;
- the reset check was reworked. `_step_cnt` now strides by `n_envs`, so the old
  `_step_cnt == reset_every + 1` equality can never hold; and a bare weight
  comparison would also trip on every ordinary gradient step. It now wraps
  `_reset_head` to count calls and confirms the head moved on the frame that
  reset.

### A3 — one assertion in the shared train loop

`stage_02_dqn/train.py`, right after `actions = agent.step(states, masks)`:

```python
assert np.shape(actions) == (num_envs,), (
    f'{type(agent).__name__}.step() returned {np.shape(actions)}, '
    f'want ({num_envs},) -- a single-env step() on a vec env')
```

Every stage's `step()` has the same `(N,64,64,3)` in, `(N,)` out contract, and
nothing checked it. This turns a confusing `IndexError` 15 frames away into an
immediate named failure. Assertion-only: no behaviour change, no effect on any
recorded number.

### What was verified locally

| Check | Result |
|---|---|
| `python stage_10c_no_per/train.py --test-no-per` | **PASS**, incl. the new width line |
| Same new test, run against the **old** `step()` | **FAIL** on `step() gave one action per env` |
| All 16 stage self-tests (04-12, incl. solo variants) | **16/16 PASS** |
| 16-env 10c run, 30k steps | **passed step 20,000** — exactly where both crashed runs died |
| 16-env 10c run, low warmup | **`loss=1.5753, meanQ=0.404`** — finite, so the learning path works at 16 envs |

The second row is the one that matters: the new test fails on the old code and
passes on the fixed code, so it is a real regression guard rather than a test
that agrees with whatever is in front of it.

One caveat on the local smoke run: the default warmup is 20,000 *agent steps*,
which at 16 envs is 320,000 env steps, so a short run never reaches a gradient
step and `loss` stays `nan` legitimately — that is why the smoke command in step
2 lowers `--warmup-steps`. With warmup at 1,000 the run reports
`loss=1.5753, meanQ=0.404` at step 5,008, i.e. gradient steps are running, the
batched forward pass over the `need_q` envs is feeding real data, the epsilon
schedule is advancing on the per-call clock (`eps=0.92`), and the new A3
contract assert passes on live 16-env data. On the cluster the real 200k run
clears the default warmup normally.

---

## 3. Cluster commands, in order

Everything goes through `./run.sh`, which collects artifacts into the committed
layout (`train_seed<N>.log`, `stats_seed<N>.jsonl`, `summary_seed<N>.json`,
`command_seed<N>.txt`, `selftests.txt`) and records the `mask` flag in the
summary. All commands were `--dry-run` verified to emit the intended `train.py`
lines.

### Step 0 — sync and sanity-check (free, ~1 min)

```bash
cd /workspace/crafter-rl        # or wherever the repo lives on the pod
git pull
nvidia-smi
bash runpod/status.sh
```

`status.sh` prints cores, RAM, GPU and every score so far. If `nvidia-smi` is
empty, stop: `sweep.sh` refuses to train on CPU on purpose, and so should you.

### Step 1 — self-tests (free, ~2 min)

```bash
python stage_10c_no_per/train.py --test-no-per
```

Expect `PASS` with:

```
step() gave one action per env     True   want True  (4 envs)
```

If it says `FAIL`, read which line is `False` and stop — do not spend GPU hours.

Optionally confirm the shared assert broke nothing else:

```bash
for t in "04 --test-nstep" "09 --test-ez" "10 --test-drq" "11 --test-aux" "12 --test-demos"; do
  set -- $t; echo "== $1"; python stage_$1*/train.py $2 2>&1 | tail -1
done
```

### Step 2 — multi-env smoke test (~5-10 min GPU)

The real check. It must get *past* 20,000 env steps — that is where the old code
died — and it needs a lowered warmup to reach a gradient step at all:

```bash
python stage_10c_no_per/train.py \
  --total-steps 40000 --num-envs 16 --seed 1 \
  --warmup-steps 1000 --buffer-size 20000 --log-every 5000 \
  --logdir /tmp/10c_smoke --eps-decay-steps 50000 --no-mask
```

Three things to confirm:

1. it reaches the `[ ...][   40000]` line with **no traceback** (the old code
   died at ~20k);
2. `loss` and `meanQ` are **finite** on the later lines, not `nan`;
3. `eps` has started falling below 1.00.

Run one at a time — parallel runs on one box halve each other's fps.

### Step 3 — the real 10c pair (~1.5-2 GPU-hours)

```bash
./run.sh 10c --envs 16 --steps 200000 --seeds 1,2 \
  --name 200k_e16_epsfix_fixed \
  -- --eps-decay-steps 50000 --no-mask
```

**Note the `--name`.** The crashed run already occupies
`results/200k_e16_epsfix`, and reusing that name would overwrite the two
tracebacks — which are the evidence of what went wrong. Keep them; use a new
directory. (If you would rather reuse the old name, commit the crash logs
somewhere first.)

To detach so an ssh drop does not kill it:

```bash
bash runpod/sweep.sh --detach --envs 16 --steps 200000 --seeds 1,2 \
  --name 200k_e16_epsfix_fixed --extra "--eps-decay-steps 50000 --no-mask" 10c
tail -f /workspace/logs/sweep-*.out
```

Heads-up: `sweep.sh` caps `--envs` at 8 when you *omit* it. Pass `--envs 16`
explicitly to match the stage 02/03 runs, or the result is not comparable.

### Step 4 — mask A/B seeds 3 and 4 (~4.8 GPU-hours)

Only if the budget allows; see the priority note below.

```bash
for st in 02 03; do
  ./run.sh $st --envs 16 --steps 200000 --seeds 3,4 --name 200k_mask \
    -- --eps-decay-steps 50000 --mask
  ./run.sh $st --envs 16 --steps 200000 --seeds 3,4 --name 200k_nomask \
    -- --eps-decay-steps 50000 --no-mask
done
```

These append `summary_seed3/4.json` beside the existing seeds 1-2 in the same
`200k_mask` / `200k_nomask` dirs, so nothing is overwritten.

### Step 5 — collect and stop the pod

```bash
bash runpod/status.sh          # score table for everything
bash runpod/sync.sh            # pull artifacts back
bash runpod/stop.sh            # the pod bills by wall clock even when idle
```

Then, locally:

```bash
git add -A && git commit -m "Fix stage 10c single-env step(); 16-env 10c results"
```

---

## 4. The mask A/B — no code change, just underpowered

Eight runs came back: 200k steps, 16 envs, eps-decay 50k, ~36 min each. I
recomputed `score_percent` from `success_percent` as the Crafter geometric mean
and it matches the reported value (3.0489 vs 3.049), so the metric is sound.

| stage | mask (s1 / s2) | mean | no-mask (s1 / s2) | mean | delta |
|---|---|---|---|---|---|
| 02 DQN | 3.049 / 2.933 | **2.991** | 3.558 / 3.428 | **3.493** | **-0.50** mask hurts |
| 03 Double | 3.894 / 3.548 | **3.721** | 3.384 / 3.115 | **3.250** | **+0.47** mask helps |

The effect flips sign between two adjacent stages at near-equal magnitude. Each
arm moved coherently rather than on one lucky achievement: stage 02 + mask
regresses `place_table` -11.9, `place_plant` -11.2, `collect_wood` -6.1,
`eat_cow` -5.2; stage 03 + mask improves `wake_up` +13.3, `collect_drink` +8.5,
`make_wood_sword` +4.0.

**Why it cannot be claimed yet.** Within-arm seed spread is the same size as the
between-arm effect: stage 03 spreads 0.346 (mask) and 0.269 (no-mask) against a
0.47 effect. For scale, stage 02's own `200k_e16_epsfix` pair — one single
configuration, two seeds — spans 3.441 / 2.975, a spread of **0.47**, exactly
the size of the effect being measured.

Two readings fit equally well: noise, or a real interaction (masking removes the
illegal actions whose Q-values vanilla DQN overestimates, so stage 02 loses a
crutch while Double-DQN, which already corrects overestimation, banks the clean
benefit). The second would be a good slide. It is also exactly the kind of story
two seeds will happily invent.

With n=4, report mean +- spread and check whether the arms' ranges still
overlap. Three outcomes: the sign flip survives (worth a slide, stated with the
spread shown); both deltas shrink toward zero (masking is score-neutral at 200k,
keep it for the legality guarantee); or the deltas stay large and same-signed
(the flip was noise). State n and the spread either way; with 4 seeds, resist a
p-value.

---

## 5. Priorities, given Nov 8 and the $40 cap

1. **Steps 0-2** — free or minutes, unblocks 10c. Do regardless.
2. **Step 3**, the 10c pair — ~1.5-2 GPU-hours. 10c currently has *zero* valid
   multi-env data.
3. **Step 4**, mask seeds 3-4 — ~4.8 GPU-hours, the expensive item.

If the budget will not cover both, **prefer 3**: 10c has no valid data at all,
whereas the mask A/B has a result that is merely unproven. "The effect reverses
between stages 02 and 03 at a magnitude matching seed noise, so we do not claim
a masking benefit at 200k" is a legitimate, honest slide that costs nothing.

### One caveat on what the 10c number will prove

The 16-env switch alone is worth roughly +0.5 score (stage 02 `200k_eps_fix`
3.128/2.735 vs `200k_e16_epsfix` 3.441/2.975), so a 16-env 10c should be
expected to beat its old 1-env 2.001/2.125 on the env-count change by itself.

To answer stage 08's actual question — does dropping PER cost anything — stage
10 and 10c must be compared **at the same env count**. Stage 10's only data
(`200k_eps_fix`, 2.017/2.034) is 1-env. So either also re-run stage 10 at 16
envs, or say plainly in the slides that the 10-vs-10c comparison is the 1-env
pair.

Nothing in this pull changes the ladder's overall shape: stages 02-03 remain
strongest (~3.0-3.9) and every stage from 05 on still underperforms plain
Double-DQN at 200k (05: 1.8-1.9, 06: 1.7, 07: 1.8, 08: 1.7-1.8, 09: 1.8-1.9,
10: 2.0, 10c: 2.0-2.1). Those later stages are *all* 1-env runs, which is very
likely part of the gap — a further argument for re-running the late ladder at 16
envs before concluding the mechanisms do not help.
