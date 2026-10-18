"""Stage 09 -- epsilon-z-greedy: committed random walks instead of twitching.

Dabney, Ostrovski & Barreto 2021, https://arxiv.org/abs/2006.01782

Crafter's map is 64x64 tiles and the agent sees 9x9. Standard epsilon-greedy
picks a fresh random action each step, so the random part of the behaviour is
a twitch in place that never travels anywhere new. Instead, when exploring,
commit to one action for a random number of steps drawn from a heavy-tailed
zeta distribution -- capped, see the comment in step().

The budget version of EDE's exploration, which runs a 5-head ensemble and
Thompson-samples epistemic uncertainty at 5x the output-layer cost.

step() only, plus two lines in end_episode().

Run:
    python stage_09_ez_greedy/train.py --test-ez
    python stage_09_ez_greedy/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import _to_tensor, main as stage_02_main
from stage_08_per.train import (PERQRDQN, build as build_08,
                                extra_args as extra_08)

STAGE_ID = 'stage_09_ez_greedy'


class EzGreedy(PERQRDQN):

    def __init__(self, *args, ez_mu=2.0, ez_max_repeat=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu, self._cap = ez_mu, ez_max_repeat
        # ONE WALK PER ENV. A repeat is "hold this action for the next k steps
        # in THIS world". Shared across num_envs envs, a walk drawn for env 0
        # would be consumed by the other envs' steps instead, so no env ever
        # holds an action for its own k steps -- which is the whole mechanism.
        self._repeat_action = np.zeros(self._num_envs, dtype=np.int64)
        self._repeat_left = np.zeros(self._num_envs, dtype=np.int64)

    def step(self, states):
        n = len(states)
        if self._step_cnt < self._warmup_steps:
            return np.random.randint(self._action_num, size=n)
        # Advance the schedule once per call, repeating or not, so epsilon
        # decays on the same clock as stages 02-08.
        self.eps = next(self._epsilon)

        actions = np.zeros(n, dtype=np.int64)
        need_q = np.zeros(n, dtype=bool)
        for i in range(n):
            if self._repeat_left[i] > 0:
                self._repeat_left[i] -= 1
                actions[i] = self._repeat_action[i]
            elif np.random.rand() < self.eps:
                a = np.random.randint(self._action_num)
                # The cap is not optional: over 200k draws zipf(2.0) has a
                # median of 1 and a p99 near 60, but a max above 300,000 --
                # one draw would hold an action for 30 episodes. Capped at 100
                # the mean is 3.8 and p99 stays near 60, so the tail survives.
                self._repeat_action[i] = a
                self._repeat_left[i] = min(int(np.random.zipf(self._mu)),
                                           self._cap) - 1
                actions[i] = a
            else:
                need_q[i] = True
        # One batched forward pass for whichever envs want a greedy action.
        if need_q.any():
            with torch.no_grad():
                q = self._q(_to_tensor(states[need_q]))
                actions[need_q] = q.argmax(1).cpu().numpy()
        return actions

    def end_episode(self, i):
        super().end_episode(i)              # stage 04 flushes its window here
        self._repeat_left[i] = 0            # a new world; drop the walk


def extra_args(p):
    extra_08(p)
    p.add_argument('--ez-mu', type=float, default=2.0)
    p.add_argument('--ez-max-repeat', type=int, default=100)
    p.add_argument('--test-ez', action='store_true',
                   help='print the zipf draw statistics and exit')


def build(args, capacity):
    net, buffer, extra = build_08(args, capacity)
    return net, buffer, dict(extra, ez_mu=args.ez_mu,
                             ez_max_repeat=args.ez_max_repeat)


# ------------------------------------------------------------- self-test


def test_ez(draws=200_000, mu=2.0, cap=100):
    """Show why the cap is not optional, and that it keeps the heavy tail.

    np.random.zipf is seeded by stage 02's reset_rng, so runs stay
    reproducible.
    """
    np.random.seed(0)
    raw = np.random.zipf(mu, draws)
    capped = np.minimum(raw, cap)
    print(f'zipf({mu}) over {draws:,} draws')
    print(f'{"":>8} {"median":>7} {"mean":>9} {"p99":>6} {"max":>9}')
    for name, v in (('raw', raw), ('capped', capped)):
        print(f'{name:>8} {np.median(v):7.0f} {v.mean():9.1f} '
              f'{np.percentile(v, 99):6.0f} {v.max():9.0f}')
    # The tail has to survive the cap, or this is just epsilon-greedy with
    # extra steps.
    tail = float((capped > 10).mean())
    print(f'\ndraws longer than 10 steps: {tail:.1%}  want > 1%')
    print(f'uncapped max is {raw.max() / 10_000:.1f}x a full 10,000-step '
          f'episode' if raw.max() > 10_000 else '')
    ok = tail > 0.01 and capped.max() <= cap and np.median(capped) == 1
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-ez' in (argv if argv is not None else sys.argv[1:]):
        return test_ez()
    return stage_02_main(argv, agent_cls=EzGreedy, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
