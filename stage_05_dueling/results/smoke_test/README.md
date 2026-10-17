# Stage 05 -- dueling head smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_05_dueling/train.py --test-dueling
python stage_05_dueling/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 1.6 min (21 fps) |
| Score | 1.55% (random baseline ~1.6%) |

Non-zero achievements: wake_up 100.0%, collect_sapling 54.5%, place_plant 54.5%, collect_wood 27.3%, place_table 9.1%, collect_drink 9.1%.

`--test-dueling` passes: output is (B, 17) and mean over actions of Q equals
V(s) to 3.7e-09, which catches a missing mean-advantage subtraction or a
dropped keepdim. Slower than stage 04 (21 vs 27 fps): at the lab's stride=1
each stream's first linear layer is 3.7M params.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
