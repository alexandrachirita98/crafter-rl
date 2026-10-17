# Stage 10b -- DrQ and resets alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** Two independent regularisers: pad-4-and-random-crop on every sampled batch, and periodic re-initialisation of the head while the buffer is kept.

This is the **isolation** row for this mechanism. The cumulative
`stage_10_drq_resets` measures the same change stacked on every earlier stage;
this file measures it against stage 03 (Double DQN) alone, so the ablation table can say
what the mechanism is worth on its own.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_10b_drq_resets_solo/train.py --test-solo
python stage_10b_drq_resets_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200  --reset-every 800
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was never consulted for a decision. The
score below is a random-policy score; reading it as "this mechanism scores
1.29%" would be wrong twice over, since the policy was not the algorithm's
and 11 episodes is not a sample.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 1.2 min (27 fps) |
| Score | 1.29% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.0135 -> 0.0603 |

Non-zero achievements: `wake_up` 100.0%, `collect_sapling` 63.6%, `place_plant` 45.5%, `collect_wood` 27.3%, `collect_drink` 9.1%.

## Evidence the mechanism is wired

Stage 03's estimator is one `nn.Sequential`, so "the head" is its two Linear
layers rather than the cumulative stage's value/adv streams. Both resets fired
and the buffer survived each:

```
step      800  RESET: re-initialised 2 head layer(s), buffer kept (800 frames)
step    1,600  RESET: re-initialised 2 head layer(s), buffer kept (1,600 frames)
```

`--test-solo` checks the shift preserves `(B,3,64,64)` uint8 and moves pixels,
that at the resetting call the head is re-initialised and the target net
re-synced, and that `_reset_head` alone leaves **all 3 convs** untouched.

**The buffer is NOT cleared** -- that is the whole trick. `--reset-every 800`
is used so a 2,000-step smoke test exercises the reset twice; the default is
100,000.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
