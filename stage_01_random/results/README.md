# Stage 01 results — random agent

Run on **local WSL2, CPU only**. Free, no RunPod. Reproduce with:

```bash
python -m common.validate_scorer                    # the gate, instant
python stage_01_random/train.py --seed 1            # 1M steps, ~32 min
```

## Headline

| | |
| :-- | :-- |
| Score | **1.51%** |
| Published baseline | **1.550% ± 0.013** over 5 seeds |
| Episodes | 5,959 (published range 5,910–5,991) |
| Mean episode length | 167.8 steps |
| Mean episode reward | 1.389 |
| Throughput | 517 fps, 32.2 min for 1M steps |

**Verdict: pass.** Our single seed scores 1.51% against a published
five-seed mean of 1.550% (individual seeds
1.54, 1.54, 1.56, 1.53, 1.57, std 0.013).

### Compare against 1.550, not against 1.6

The "1.6%" in `README.md` is `1.550` **rounded to one decimal for display**.
Scoring the shipped `scores/crafter_noreward-random.json` per seed gives
1.550 ± 0.013. Treating 1.6 as the target makes a correct
implementation look 0.09 points off. The honest statement is
**1.51% against 1.550 ± 0.013**.

Note the published file is `crafter_noreward-*`: the baseline was collected with
rewards disabled. For a uniform-random policy that changes no behaviour, which is
why the per-achievement rates below line up.

## Per-achievement success rate

| achievement | ours | published mean |
| :-- | --: | --: |
| `wake_up` | 93.45% | 93.60% |
| `collect_sapling` | 51.85% | 50.19% |
| `place_plant` | 46.60% | 44.62% |
| `collect_wood` | 23.81% | 24.38% |
| `collect_drink` | 9.60% | 9.32% |
| `place_table` | 3.99% | 4.36% |
| `eat_cow` | 0.32% | 0.38% |
| `make_wood_pickaxe` | 0.20% | 0.32% |
| `make_wood_sword` | 0.18% | 0.32% |
| `defeat_zombie` | 0.08% | 0.14% |
| `defeat_skeleton` | 0.00% | 0.02% |
| `collect_stone` | 0.00% | 0.01% |
| `place_stone` | 0.00% | 0.01% |
| `eat_plant` | 0.00% | 0.01% |

The remaining 8 achievements are 0% in both, up through the iron and
diamond tier. Nothing above `place_table` is reachable by chance — that is the
gap every later stage has to close.

The four rows where we read 0.00% against a published 0.01–0.02% are events of
roughly one episode in 5,000–10,000. At 5,959 episodes that is a coin
flip, not a discrepancy.

## Scorer gate

The real purpose of this stage. `common/validate_scorer.py` scores the seven
published runs in `scores/` with **our** `common/metrics.py` and compares them to
the scoreboard in `README.md`. Output in `validate_scorer.txt`:

```
method                        ours  published   delta
------------------------------------------------------
dreamerv2                    9.98%      10.0%  -0.02  OK
human                       50.53%      50.5%  +0.03  OK
ppo                          4.57%       4.6%  -0.03  OK
rainbow                      4.31%       4.3%  +0.01  OK
random                       1.55%       1.6%  -0.05  OK
unsup_plan2explore           2.11%       2.1%  +0.01  OK
unsup_rnd                    2.01%       2.0%  +0.01  OK

All 7 scored methods match the published table within 0.1 points. The scorer is trustworthy.
```

Rainbow at **4.31%** against a published 4.3% is the number that matters. It means
a later claim of "ours 4.8% vs Rainbow 4.3%" has both figures coming out of the
same function, rather than comparing two codebases by eye.

Note the `random` row there reads 1.55% — that is our scorer applied to *their*
data, an independent check on the scoring function. Our agent's
1.51% is the separate check on the environment setup. Both pass.

## Carry forward

**517 fps single-threaded** means 1M env steps costs 32.2 minutes of
irreducible CPU time per seed per stage, before any learning. Pod selection should
follow vCPU count over GPU size.

**Two near-miss numbers, both from recall instead of reading.** The validator first
expected DreamerV2 at 13.7% and failed — that is DreamerV3's score; DreamerV2 is
10.0%. Same class of error as 1.6 versus 1.550. In both cases the code was right
and the constant was wrong, which is the expensive direction to get confused in
when the number ends up on a slide.

## Files

| File | What |
| :-- | :-- |
| `summary_seed1.json` | every number above, machine-readable |
| `train_seed1.log` | full stdout, all 5,959 episode lines |
| `stats_seed1.jsonl` | raw per-episode records from `crafter.Recorder` |
| `validate_scorer.txt` | scorer gate output |
