# Stage 10c -- full stack minus PER smoke test -- 2,000 steps

Wiring test only: does it run end to end without crashing. Epsilon barely
moves off 1.0 in 2,000 steps, so every action is random and the score is a
random-policy score. No learning is expected or measured here.

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_10c_no_per/train.py --test-no-per
python stage_10c_no_per/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000  --reset-every 1000
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 11 |
| Wall time | 1.6 min (21 fps) |
| Score | 0.99% (random baseline ~1.6%) |

Non-zero achievements: collect_sapling 63.6%, place_plant 54.5%,
wake_up 54.5%, collect_wood 18.2%.

`--test-no-per` passes. It checks the splice rather than the mechanisms, each
of which is tested in the stage that introduces it: the buffer is the uniform
n-step one and `sample()` returns 6 tensors not 8, `PERQRDQN` is out of the
MRO, `_update` resolves to QR-DQN's six-argument version and
`_target_quantiles` to Munchausen's, no `total_steps` reaches the agent
(there is no beta to anneal), and stages 09 and 10 both survive the swap --
the committed walk runs and the head reset fires with the buffer kept.

Run with `--reset-every 1000`, same as stage 10, so a 2,000-step smoke test
exercises the reset twice; the default is 100,000.

## Against stage 10

Same flags, same seed, same 11 episodes. 21 fps vs stage 10's 16 fps, which
is the sum-tree and the priority updates not happening -- the only expected
difference at this budget. The scores (0.99% vs 0.72%) are both
random-policy noise over 11 episodes and say nothing either way; the PER
question is for 200k x 2 seeds on RunPod, not for this.

## Files

- `train_seed1.log` -- full stdout of the run
- `stats_seed1.jsonl` -- per-episode stats from the run's logdir
