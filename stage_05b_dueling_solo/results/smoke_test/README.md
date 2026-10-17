# Stage 05b -- dueling head alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** Only the network changes: the head splits into V(s) and A(s,a) with the mean-advantage subtraction. The agent, buffer, loss and target are stage 03's.

This is the **isolation** row for this mechanism. The cumulative
`stage_05_dueling` measures the same change stacked on every earlier stage;
this file measures it against stage 03 (Double DQN) alone, so the ablation table can say
what the mechanism is worth on its own.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_05b_dueling_solo/train.py --test-solo
python stage_05b_dueling_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was never consulted for a decision. The
score below is a random-policy score; reading it as "this mechanism scores
1.19%" would be wrong twice over, since the policy was not the algorithm's
and 12 episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 12 |
| Wall time | 1.6 min (20 fps) |
| Score | 1.19% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.0038 -> 0.0354 |

Non-zero achievements: `wake_up` 100.0%, `collect_sapling` 66.7%, `place_plant` 50.0%, `collect_drink` 8.3%, `collect_wood` 8.3%.

## Evidence the mechanism is wired

`--test-solo` confirms the isolation holds: the agent is `DoubleDQN` with no
n-step machinery, the buffer is stage 02's one-step `ReplayMemory`, no extra
constructor kwargs are passed, and `mean_a Q(s,a) == V(s)` to 1e-5 -- the
check that catches a missing mean-advantage subtraction.

Throughput 20 fps against stage 03's 27: at the lab's `stride=1` each stream's
first linear layer is 3.7M parameters, so the dueling net is roughly twice the
size of stage 03's.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
