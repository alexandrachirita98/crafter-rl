# Stage 08b -- prioritised replay alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** Uniform sampling becomes P(i) proportional to |TD error|^alpha via a sum tree, with importance-sampling correction and beta annealed 0.4 -> 1.0.

This is the **isolation** row for this mechanism. The cumulative
`stage_08_per` measures the same change stacked on every earlier stage;
this file measures it against stage 03 (Double DQN) alone, so the ablation table can say
what the mechanism is worth on its own.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_08b_per_solo/train.py --test-solo
python stage_08b_per_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was never consulted for a decision. The
score below is a random-policy score; reading it as "this mechanism scores
0.89%" would be wrong twice over, since the policy was not the algorithm's
and 14 episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 14 |
| Wall time | 1.3 min (25 fps) |
| Score | 0.89% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.0088 -> 0.0086 |

Non-zero achievements: `wake_up` 78.6%, `collect_sapling` 42.9%, `place_plant` 42.9%, `collect_wood` 7.1%.

## Evidence the mechanism is wired

The cumulative stage 08 prioritises by the per-sample **quantile** loss, which
only exists once stage 06 is in. This variant prioritises by **|TD error|** on
stage 03's scalar target -- Schaul's original formulation, which needs nothing
above Double DQN.

`--test-solo` confirms no n-step and no quantiles are inherited, then checks
the tree on the five-tuple buffer: **40/40 transitions stored** (the
cumulative version holds 2 back in its n-step window), every written leaf
above 0, all 24 power-of-two padding leaves exactly 0 and therefore
unreachable, and beta annealed to 0.97.

**Expect nothing from this.** contribution.md section 4.2 notes EDE deleted
prioritised replay and got 2.7x better. The isolated row exists so a null
result can be attributed to PER itself rather than to PER-on-top-of-five-other-things.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
