# Stage 03 smoke test -- Double DQN, 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here, and the
difference from stage 02 is seed noise, not the double target.

Run on local WSL2, CPU-only torch (2.13.0+cpu), 2026-10-17.

```bash
python stage_03_double/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 1.6 min (21 fps) |
| Score | 1.64% (random baseline ~1.6%) |

Non-zero achievements: wake_up 100%, collect_sapling 72.7%, place_plant 63.6%,
collect_wood 36.4%, collect_drink 9.1%, place_table 9.1%.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats copied from
  `logdir/crafter_reward-stage_03_double/1/stats.jsonl`
