# Stage 09 -- epsilon-z-greedy smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_09_ez_greedy/train.py --test-ez
python stage_09_ez_greedy/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 12 |
| Wall time | 1.9 min (18 fps) |
| Score | 0.94% (random baseline ~1.6%) |

Non-zero achievements: wake_up 66.7%, collect_sapling 41.7%, place_plant 41.7%, collect_wood 16.7%.

`--test-ez` passes: over 200,000 draws zipf(2.0) has median 1 and p99 60 but
a max of 31,462 -- 3.1x a full episode, which is why the cap is not optional.
Capped at 100 the mean is 3.7, p99 stays 60 and 5.8% of draws still exceed 10
steps, so the heavy tail survives.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
