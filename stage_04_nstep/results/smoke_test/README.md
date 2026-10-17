# Stage 04 -- n-step returns smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_04_nstep/train.py --test-nstep
python stage_04_nstep/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 13 |
| Wall time | 1.2 min (27 fps) |
| Score | 1.38% (random baseline ~1.6%) |

Non-zero achievements: wake_up 84.6%, collect_sapling 76.9%, place_plant 53.8%, collect_wood 30.8%, collect_drink 15.4%.

`--test-nstep` passes: the scripted five-step episode (rewards 1 0 0 1 0,
death last, gamma 0.9, n 3) pushes the five expected n-step tuples, and the
cap flush after `end_episode()` yields n = 2, 1 with mask still 1.0.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
