# Stage 08 -- prioritised replay smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_08_per/train.py --test-per
python stage_08_per/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 1.9 min (17 fps) |
| Score | 1.37% (random baseline ~1.6%) |

Non-zero achievements: wake_up 100.0%, collect_sapling 72.7%, place_plant 63.6%, collect_wood 18.2%, collect_drink 18.2%.

`--test-per` passes: every written leaf carries a priority above 0 (range
1.00-2.88), all 26 power-of-two padding leaves stay exactly 0 and therefore
unreachable, and beta annealed to 0.97 over a 40-step run.

A null result at 200k x 2 seeds is the expected outcome here, not a bug --
see the module docstring and contribution.md section 4.2.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
