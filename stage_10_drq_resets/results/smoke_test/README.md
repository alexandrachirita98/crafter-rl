# Stage 10 -- DrQ and periodic resets smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_10_drq_resets/train.py --test-drq
python stage_10_drq_resets/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000  --reset-every 1000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 2.0 min (16 fps) |
| Score | 0.72% (random baseline ~1.6%) |

Non-zero achievements: wake_up 72.7%, collect_sapling 54.5%, place_plant 36.4%.

`--test-drq` passes: the shift preserves (B,3,64,64) uint8 and moves pixels;
at the reset step the adv head is re-initialised, the target net is re-synced,
`_reset_head` alone leaves the trunk untouched, and the buffer keeps all its
transitions.

Run with `--reset-every 1000` so a 2,000-step smoke test actually exercises
the reset twice; the default is 100,000.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
