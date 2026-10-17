# Stage 07 -- Munchausen smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_07_munchausen/train.py --test-munchausen
python stage_07_munchausen/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 10 |
| Wall time | 1.7 min (19 fps) |
| Score | 1.36% (random baseline ~1.6%) |

Non-zero achievements: wake_up 100.0%, collect_sapling 60.0%, place_plant 60.0%, collect_wood 20.0%, collect_drink 20.0%.

`--test-munchausen` passes: the bonus broadcasts over all 51 quantiles,
stays inside [-1, 0], and 12 of 16 rows sit exactly on the clip, so the clip
is wired in. The soft value (1.894) stays under max_a Q + tau*log(17) = 1.976
for every row.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
