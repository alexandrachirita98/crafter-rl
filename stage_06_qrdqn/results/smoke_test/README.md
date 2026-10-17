# Stage 06 -- QR-DQN smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_06_qrdqn/train.py --test-loss
python stage_06_qrdqn/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 12 |
| Wall time | 1.5 min (22 fps) |
| Score | 1.45% (random baseline ~1.6%) |

Non-zero achievements: wake_up 75.0%, collect_sapling 50.0%, place_plant 41.7%, collect_wood 25.0%, place_table 8.3%, collect_drink 8.3%.

`--test-loss` passes: fitting 8 quantiles to N(5,3) recovers them to within
0.41, the mean of the fitted quantiles is 4.98 against a true 5.00, and they
come out monotone increasing (a transposed loss breaks monotonicity).

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
