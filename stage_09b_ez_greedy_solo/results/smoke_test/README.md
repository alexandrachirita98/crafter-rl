# Stage 09b -- epsilon-z-greedy alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** Only action selection changes: when exploring, commit to one random action for a zeta-distributed number of steps instead of redrawing every step.

This is the **isolation** row for this mechanism. The cumulative
`stage_09_ez_greedy` measures the same change stacked on every earlier stage;
this file measures it against stage 03 (Double DQN) alone, so the ablation table can say
what the mechanism is worth on its own.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_09b_ez_greedy_solo/train.py --test-solo
python stage_09b_ez_greedy_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was never consulted for a decision. The
score below is a random-policy score; reading it as "this mechanism scores
1.15%" would be wrong twice over, since the policy was not the algorithm's
and 10 episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 10 |
| Wall time | 1.2 min (28 fps) |
| Score | 1.15% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.0027 -> 0.0463 |

Non-zero achievements: `wake_up` 70.0%, `collect_drink` 40.0%, `collect_sapling` 30.0%, `place_plant` 20.0%, `defeat_skeleton` 10.0%.

## Evidence the mechanism is wired

This is the cleanest isolation in the set -- the change is purely behavioural,
so the network, buffer, loss and target are all stage 03's and any difference
from stage 03 is the walk.

The run reports the mechanism directly:

```
ez-greedy: 413 exploratory draws, 167 committed, mean drawn duration 3.70,
longest 100 (cap 100), 1,087 steps (54.4% of the run) inside a commitment
```

`--test-solo` drives the real `step()` at eps=1.0 for 20,000 steps: 5,252
draws, mean duration 3.81, 73.7% of steps inside a commitment, longest single
draw 100, and `end_episode()` clears an in-flight walk.

It measures the longest **draw**, not the longest run of identical actions.
Adjacent draws collide on the same action 1 in 17 times, so observed identical
runs legitimately exceed the cap (116 was seen) while every individual draw
respects it.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
