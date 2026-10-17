# Stage 06b -- QR-DQN alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** The head emits 51 quantiles per action instead of one mean, and quantile Huber regression replaces MSE. No dueling split, no n-step window -- the target is one-step.

This is the **isolation** row for this mechanism. The cumulative
`stage_06_qrdqn` measures the same change stacked on every earlier stage;
this file measures it against stage 03 (Double DQN) alone, so the ablation table can say
what the mechanism is worth on its own.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_06b_qrdqn_solo/train.py --test-solo
python stage_06b_qrdqn_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was never consulted for a decision. The
score below is a random-policy score; reading it as "this mechanism scores
1.34%" would be wrong twice over, since the policy was not the algorithm's
and 10 episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 10 |
| Wall time | 1.2 min (27 fps) |
| Score | 1.34% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.1586 -> 0.0689 |

Non-zero achievements: `wake_up` 100.0%, `collect_sapling` 80.0%, `place_plant` 70.0%, `collect_wood` 20.0%, `collect_drink` 10.0%.

## Evidence the mechanism is wired

`--test-solo` checks the isolation (no n-step, no quantile-dueling split, a
plain one-step buffer, output `(B, 17, N)`) and the one piece of code that is
genuinely new here, the **one-step** quantile target: with `mask = 0` the
target is exactly the reward broadcast over every quantile, and with
`mask = 1` it is exactly `reward + gamma * Q(s')` -- a plain gamma, not
`gamma ** n`.

In-run loss 0.069 is the right order for a quantile loss summed over 51
prediction quantiles; an axis swap would show up here as a ~51x shift.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
