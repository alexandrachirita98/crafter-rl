"""Stage 01 -- random agent: 17 actions uniformly at random, writes stats.jsonl.

It exists to validate the scorer before there is an agent to score. This agent
should land near the published Random row (1.6%), and our scorer on the
shipped scores/*.json should reproduce Rainbow at 4.31%. The second is the
real gate -- `python -m common.validate_scorer` -- so that "ours 4.8% vs
Rainbow 4.3%" is two numbers out of one function.

Run:
    python stage_01_random/train.py --total-steps 1000000 --seed 1
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, make_env           # noqa: E402
from common.metrics import score_run                 # noqa: E402

STAGE_ID = 'stage_01_random'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults are the real experiment, per assignment_v0.pdf section 2.1:
    # bare `python train.py` must start the run we actually report.
    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps, the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--logdir', type=str, default='logdir')
    args = p.parse_args()

    logdir = (pathlib.Path(args.logdir) / f'crafter_reward-{STAGE_ID}'
              / str(args.seed))
    logdir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed, logdir)
    rng = np.random.default_rng(args.seed)

    print(f'stage {STAGE_ID}  seed {args.seed}  steps {args.total_steps:,}')
    print(f'logging to {logdir}/stats.jsonl')

    env.reset()
    ep_return, ep_len, episodes = 0.0, 0, 0
    start = time.time()
    for step in range(args.total_steps):
        # The whole policy. Uniform over the 17 actions.
        action = int(rng.integers(N_ACTIONS))
        _, reward, done, info = env.step(action)
        ep_return += reward
        ep_len += 1
        if done:
            episodes += 1
            unlocked = sum(v > 0 for v in info['achievements'].values())
            fps = (step + 1) / (time.time() - start)
            print(f'step {step + 1:>8,}  ep {episodes:>4}  '
                  f'return {ep_return:6.2f}  len {ep_len:>5}  '
                  f'ach {unlocked:>2}/22  {fps:>5.0f} fps')
            env.reset()
            ep_return, ep_len = 0.0, 0

    elapsed = time.time() - start
    print(f'\ndone: {episodes} episodes in {elapsed / 60:.1f} min '
          f'({args.total_steps / elapsed:.0f} fps)')

    stats = logdir / 'stats.jsonl'
    if stats.exists() and episodes:
        score, percents, n = score_run(stats)
        print(f'score over {n} episodes: {score:.2f}%   (published Random: 1.6%)')
        print('\nper-achievement success rate, non-zero only:')
        from common.metrics import ACHIEVEMENTS
        for name, pct in sorted(zip(ACHIEVEMENTS, percents),
                                key=lambda kv: -kv[1]):
            if pct > 0:
                print(f'  {name:<20} {pct:5.1f}%')


if __name__ == '__main__':
    main()
