# Stage 10c -- full stack minus PER -- cluster smoke test, 2,000 steps

First run of this stage on the UPB NCIT cluster GPU, via `./run.sh 10c --smoke`.
Same wiring test as `../smoke_test`, but on real CUDA instead of local CPU torch.

Wiring only: epsilon moved 1.00 -> 0.99 over 2,000 steps, so every action was
uniform random. The score is a random-policy score over 12 episodes. No
learning is expected or measured here.

Run on `ucsx-ncit-gpu-wn100`, device `cuda`, 2026-10-17.

```bash
./run.sh 10c --smoke
```

## Result

| | |
| :-- | :-- |
| Exit code | 0 |
| Episodes | 12 |
| Wall time | 0.5 min (69 fps) |
| Score | 1.11% (random baseline ~1.6%) |
| Loss (first -> last logged) | 1.2165 -> 1.2594 |

Non-zero achievements: `wake_up` 75.0%, `collect_wood` 41.7%,
`collect_sapling` 25.0%, `place_plant` 16.7%, `collect_drink` 8.3%.

`--test-no-per` passes -- all 10 checks, same as the local run.

The first two logged intervals show `loss=nan, meanQ=nan`: that is warmup,
`--warmup-steps 500` means no gradient step has run yet by step 400.

## Against the local CPU run

Same stage, same seed, same flags, different machine.

| | local WSL2 (CPU) | cluster (cuda) |
| :-- | :-- | :-- |
| fps | 21 | 69 |
| Wall time | 1.6 min | 0.5 min |
| Episodes | 11 | 12 |
| Score | 0.99% | 1.11% |

**3.3x throughput** is the point of this run -- the GPU is doing the
gradient steps. Episode count and score differ despite the same seed, which
is expected: device-dependent nondeterminism in the conv backward changes the
network, and with eps ~ 1.0 the policy is random either way. Both scores are
noise over ~12 episodes and neither says anything about the algorithm.

fps decays within the run (139 -> 69) as the replay buffer fills and every
step starts doing a gradient update.

## Files

- `train_seed1.log` -- full stdout, including the `--test-no-per` block
- `selftests.txt` -- `--test-no-per` output on its own
- `command_seed1.txt` -- the exact commands `run.sh` issued
- `summary_seed1.json` -- headline numbers, transcribed

`stats_seed1.jsonl` was written on the cluster but is **not** copied here --
its contents were not in the captured terminal output. Re-fetch from the
cluster at `stage_10c_no_per/results/live_run/stats_seed1.jsonl` if the
per-episode records are needed.
