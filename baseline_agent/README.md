# Coding-agent Crafter baseline — PPO

**This code was written by a coding agent (Claude Code), not by hand.** It
exists only to produce the reference score that the hand-written agent in
`agent/` must beat, per the assignment. Do not submit it as your own work.

## Usage

```sh
python baseline_agent/train.py                       # 1e6 steps, defaults
python baseline_agent/train.py --steps 2000 --outdir /tmp/smoke   # smoke test
```

Episode stats land in `<outdir>/shard<N>/stats.jsonl`, one JSON object per
episode, written by `crafter.Recorder(save_stats=True)`. Checkpoints go to
`<outdir>/ckpt.pt` every 100k env steps.

## Why PPO and not DQN

The brief allowed either. PPO was chosen for one dominating reason on this
machine: **memory**. A DQN replay buffer of 10⁶ frames at 64×64×3 is 12 GB as
uint8 — and this station has 7.4 GB of RAM total, ~2 GB free. Capping the
buffer at 10⁵ would have been possible but costs DQN most of its sample
efficiency. PPO's on-policy rollout buffer is bounded by the horizon instead:

```
16 envs × 64 steps × 64×64×3 uint8 = 12 MB
```

Three orders of magnitude smaller, and it leaves the 1M-step budget as the only
real constraint. PPO also parallelises across 16 CPU cores, which matters here
because torch is a CPU-only wheel (`2.13.0+cpu`, `cuda.is_available() == False`).

## Hyperparameters

| Parameter | Value | Reasoning |
| :-- | :-- | :-- |
| `--num-envs` | 16 | Matches the 16 cores; batch of 1024 per update. |
| `--horizon` | 64 | Short rollouts → 976 updates within 1M steps. |
| `--gamma` | 0.95 | Crafter rewards are dense-ish (±0.1 health, +1 achievement) and episodes run ~170–10000 steps. The Crafter paper's own PPO uses a short effective horizon; 0.95 ≈ 20-step lookahead, enough for the early tech tree. |
| `--gae-lambda` | 0.65 | Low λ trades variance for bias; with only ~1k updates, variance is the binding problem. |
| `--lr` | 3e-4 | Adam, `eps=1e-5`. Standard PPO default, untuned. |
| `--epochs` | 3 | 3 passes × 4 minibatches = 12 gradient steps per rollout. |
| `--clip-coef` | 0.2 | Standard. |
| `--entropy-coef` | 0.01 | Non-zero is essential: with 17 actions and sparse achievements, a collapsed policy never discovers the tech tree. |
| `--value-coef` | 0.5 | Standard. |
| `--max-grad-norm` | 0.5 | Standard. |

Hyperparameters are **not tuned** — they are standard PPO defaults with a
Crafter-appropriate discount. A tuned baseline would be a stronger target, but
the assignment wants a defensible reference number, not a competitive one.

Network: Nature-CNN adapted to 64×64 input (conv 8/4 → 4/2 → 3/1, then 512-unit
dense), with separate linear policy and value heads. Orthogonal init; policy
head gain 0.01 so the initial policy is near-uniform.

## Implementation notes

- **Observations stay uint8** in the rollout buffer and are cast to float and
  divided by 255 in exactly one place, `ActorCritic.features()`.
- **No bootstrapping across episode boundaries.** `compute_gae` zeroes both the
  value bootstrap and the advantage carry at every `done`. This is the classic
  Crafter/PPO bug and is unit-tested — see "Verification" below.
- **The step counter counts environment steps**, incremented by `num_envs` per
  vector step, so the 1M budget is real and comparable to the random agent.
- **Per-env shard directories.** Each of the 16 envs gets its own Recorder at
  `<outdir>/shard<N>/`. A single shared `stats.jsonl` would interleave writes
  from 16 processes' worth of episodes, and `analysis/read_metrics.py` globs
  `**/stats.jsonl` treating each file as one run — the same layout the existing
  random-agent runs in `logdir/` already use.
- **Auto-reset seam.** When an env finishes, `VecEnv.step` returns the first
  observation of the *next* episode alongside `done=True`, so the terminal
  observation never enters the buffer. This is the standard vectorised
  convention and is harmless: `done` zeroes the bootstrap, so that observation
  would only have fed a value estimate multiplied by zero.
- Crafter's `step()` returns a **4-tuple** `(obs, reward, done, info)`; the code
  matches that, not gymnasium's 5-tuple.

## Scoring with 16 shards

`analysis/read_metrics.py` skips any `stats.jsonl` whose summed episode lengths
are below `budget - 1e4`. With 16 shards each holding ~1M/16 ≈ 62.5k steps,
every shard is "incomplete" at the default `budget=int(1e6)` and all 16 would be
silently dropped. When scoring, pass a per-shard budget:

```sh
python3 read_metrics.py   # with budget=int(1e6/16) for this run
```

or aggregate the shards into one list of episodes first — the one-off snippet in
`docs/running-a-coding-agent.md` §Step 8 does this and needs no budget flag.

## Verification performed

Smoke test, 4 envs × 8192 env steps, CPU, ~29 s:

```
device=cpu envs=4 horizon=64 batch=256 actions=17
budget=8000 env steps -> 31 updates
step     2048  reward   1.99  length  163.4  episodes   9  fps    335  elapsed      6s
step     4096  reward   2.62  length  162.7  episodes  23  fps    322  elapsed     12s
step     6144  reward   2.60  length  167.0  episodes  34  fps    203  elapsed     23s
step     8192  reward   2.54  length  165.1  episodes  48  fps    309  elapsed     29s
done: 8192 env steps in 29s
```

48 episodes logged across shards, matching the 48 counted by the progress
line — no episodes dropped. Each record carries all 22 `achievement_*` keys plus
`length` and `reward`.

The default 16-env smoke test at `--steps 2000` completes in **10 s** but logs 0
episodes: 2048/16 = 128 steps per env, shorter than a ~170-step random episode.
That is expected, not a failure; the 4-env run above is what exercises the
logging path.

GAE done-handling is unit-tested directly (3 steps, `done` at t=1, deliberately
huge bootstrap value of 100):

```
adv      [1.9, 1.0, 91.0]
expected [1.9, 1.0, 91.0]
PASS: no bootstrap across done, no advantage carry across done
```

Grep checks: no `stable_baselines3` / `rllib` / `sb3` / `ray.` imports, exactly
one `/255.0`, no TODOs or dead code.

**Not yet verified: the 1M-step score.** That is Step 7 of
`docs/running-a-coding-agent.md`. On CPU at ~500 steps/s a 1M-step seed is ~35
minutes; expect 4–6 % for a competent PPO versus 1.54–1.57 % random (measured in
`logdir/crafter_noreward-random/`). A score below random means a bug, not a hard
environment.
