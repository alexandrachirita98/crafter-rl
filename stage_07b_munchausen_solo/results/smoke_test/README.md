# Stage 07b -- Munchausen alone -- isolation variant, 2,000 steps

**Built on stage 03 (Double DQN).** Adds `alpha*tau*log pi(a|s)` to the reward
and replaces the Double-DQN argmax with a soft bootstrap under pi.

This is **M-DQN, the scalar method of the paper's section 3** -- the base
version, not the M-IQN variant the cumulative `stage_07_munchausen`
implements. The bonus and the soft value are identical in form; they act on
one Q-value per action instead of 51 quantiles. That keeps this row a true
single-mechanism isolation against stage 03, like every other solo variant.

Note the soft bootstrap *replaces* the argmax rather than adding to it, so the
honest reading is "Munchausen vs stage 03", not "Munchausen added to stage 03".

Run on local WSL2, CPU-only torch, 2026-10-17.

```bash
python stage_07b_munchausen_solo/train.py --test-solo
python stage_07b_munchausen_solo/train.py --total-steps 2000 --warmup-steps 500 \
    --buffer-size 5000 --log-every 200
```

## What this run is and is not

**A wiring test, not a measurement.** `--eps-decay-steps` is 250,000, so
epsilon moved 1.00 -> 0.99 and **every action was uniform random**. The
network took ~750 gradient steps but was consulted for a decision about nine
times. The score below is a random-policy score.

## Headline

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 12 |
| Wall time | 1.2 min (27 fps) |
| Score | 1.34% (random baseline ~1.6%) |
| Loss (first -> last logged) | 0.0020 -> 0.0041 |

Non-zero achievements: `wake_up` 100.0%, `collect_sapling` 66.7%, `place_plant` 58.3%, `collect_wood` 33.3%, `collect_drink` 8.3%.

## Evidence the mechanism is wired

`--test-solo` confirms the isolation (inherits from `DoubleDQN`, no n-step,
no quantiles) and checks three things about the target:

- **The clip is live.** With `mask = 0` the target is the bonus alone, every
  row inside [-1, 0], and 11 of 16 rows sitting exactly on the clip. If none
  did, the clip would not be wired in.
- **The soft value respects the entropy bound.** 3.521 against
  `max_a Q + tau*log(17) = 3.606`, for every row.
- **The bootstrap really is an expectation.** At `tau = 0.03` the softmax is
  numerically a *hard argmax* once the Q spread exceeds ~0.3, so at the
  default scale the soft value equals `max_a Q` to six decimals -- that is the
  paper's intent, the entropy term being a small regulariser rather than a
  smoothed bootstrap. Asserting `soft < max_a Q` there would fail on correct
  code. The test therefore checks it in the regime where it is
  distinguishable: with a Q spread of `tau/3`, the soft value sits strictly
  above `max_a Q`.

## Files

- `train_seed1.log` -- full stdout, with a `--log-every 200` trace
- `stats_seed1.jsonl` -- per-episode records, one line per episode
- `selftests.txt` -- `--test-solo` output
