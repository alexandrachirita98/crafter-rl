"""Stage 09b -- epsilon-z-greedy alone, on stage 03.

Exploration is the one change that is purely behavioural, so it isolates onto
stage 03 with no reimplementation. Only step() and end_episode() change --
network, buffer, loss and target are stage 03's -- so any difference from
stage 03 is the walk.

Run:
    python stage_09b_ez_greedy_solo/train.py --test-solo
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (N_ACTIONS, ReplayMemory, _to_tensor,
                                get_estimator, main as stage_02_main)
from stage_03_double.train import DoubleDQN

STAGE_ID = 'stage_09b_ez_greedy_solo'


class EzGreedyDoubleDQN(DoubleDQN):
    """Stage 03 that commits to a random action for a zeta-drawn duration."""

    def __init__(self, *args, ez_mu=2.0, ez_max_repeat=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu, self._cap = ez_mu, ez_max_repeat
        self._repeat_action = np.zeros(self._num_envs, dtype=np.int64)
        self._repeat_left = np.zeros(self._num_envs, dtype=np.int64)
        # instrumentation, for the log line and the README
        self.draws = self.committed = self.committed_steps = 0
        self.repeat_mean = 0.0
        self.longest_draw = 0

    def step(self, states):
        n = len(states)
        if self._step_cnt < self._warmup_steps:
            return np.random.randint(self._action_num, size=n)
        # Advance the schedule every step, repeating or not, so epsilon decays
        # on the same clock as stage 03.
        self.eps = next(self._epsilon)
        if self._repeat_left[0] > 0:
            self._repeat_left[0] -= 1
            self.committed_steps += 1
            return np.array([self._repeat_action[0]])
        if torch.rand(1).item() < self.eps:
            self._repeat_action[0] = torch.randint(self._action_num, (1,)).item()
            # The cap is not optional: uncapped zipf(2.0) draws exceed
            # 300,000, holding one action for 30 episodes. Measurements in
            # stage_09_ez_greedy's --test-ez.
            drawn = min(int(np.random.zipf(self._mu)), self._cap)
            self._repeat_left[0] = drawn - 1
            self.draws += 1
            if drawn > 1:
                self.committed += 1
            self.repeat_mean += (drawn - self.repeat_mean) / self.draws
            self.longest_draw = max(self.longest_draw, drawn)
            return np.array([self._repeat_action[0]])
        with torch.no_grad():
            return self._estimator(
                _to_tensor(states)).argmax(1).cpu().numpy()

    def end_episode(self, i):
        super().end_episode(i)
        self._repeat_left[i] = 0            # a new world; drop the walk


AGENT_CLS = EzGreedyDoubleDQN


def extra_args(p):
    p.add_argument('--ez-mu', type=float, default=2.0)
    p.add_argument('--ez-max-repeat', type=int, default=100)
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation and the commitment logic, then exit')


def build(args, capacity):
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {'ez_mu': args.ez_mu,
                         'ez_max_repeat': args.ez_max_repeat}


# ------------------------------------------------------------- self-test


def test_solo(steps=20_000, capacity=2000, batch=8):
    """Isolation, plus the two things the cumulative --test-ez cannot check.

    --test-ez measures the zipf distribution in the abstract. This drives the
    real step() at eps=1.0 and checks the commitments actually happen, never
    exceed the cap, and are cleared at an episode boundary.

    Note it checks the longest DRAW, not the longest run of identical actions.
    Adjacent draws collide on the same action 1 in 17 times, so observed runs
    pass 100 (116 was seen here) while every individual draw respects the cap.
    """
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import get_epsilon_schedule, reset_rng
    from stage_04_nstep.train import NStepDoubleDQN
    from stage_06_qrdqn.train import QRDQN

    no_nstep = not issubclass(AGENT_CLS, NStepDoubleDQN)
    no_quant = not issubclass(AGENT_CLS, QRDQN)
    print(f'agent carries no n-step        {no_nstep}   want True')
    print(f'agent carries no quantiles     {no_quant}   want True')

    reset_rng(0)
    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              ez_mu=2.0, ez_max_repeat=100)
    net, buffer, extra = build(args, capacity)
    agent = AGENT_CLS(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                      get_epsilon_schedule(1.0, 1.0, steps),   # eps pinned 1.0
                      N_ACTIONS, gamma=0.99, warmup_steps=batch + 1,
                      update_steps=10 ** 9, update_target_steps=10 ** 9,
                      **extra)
    agent._step_cnt = batch + 1            # past warmup, no gradient steps

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(steps):
        agent.step(frame[None])

    pct = 100.0 * agent.committed_steps / steps
    print(f'\nat eps=1.0 over {steps:,} steps:')
    print(f'  draws {agent.draws:,}, of which committed (>1 step) '
          f'{agent.committed:,}')
    print(f'  mean drawn duration {agent.repeat_mean:.2f}   '
          f'(capped zipf(2.0) mean is ~3.7)')
    print(f'  steps inside a commitment {agent.committed_steps:,} ({pct:.1f}%)')
    print(f'  longest single draw {agent.longest_draw} (cap {agent._cap})')
    fired = agent.committed > 0 and pct > 20
    # Measure the draw, not the observed run of identical actions: two
    # adjacent draws pick the same action 1 in 17 times, so identical runs
    # legitimately exceed the cap while no single draw does.
    capped = agent.longest_draw <= agent._cap
    print(f'  commitments fired     {fired}   want True')
    print(f'  cap never exceeded    {capped}   want True')

    # an episode boundary must drop an in-flight walk
    agent._repeat_left[0], agent._repeat_action[0] = 50, 3
    agent.end_episode(0)
    cleared = agent._repeat_left[0] == 0
    print(f'\nend_episode clears the walk    {cleared}   want True')

    ok = no_nstep and no_quant and fired and capped and cleared
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def report_ez(agent, total_steps):
    """The evidence the mechanism fired, which the score cannot show."""
    pct = 100.0 * agent.committed_steps / max(1, total_steps)
    print(f'ez-greedy: {agent.draws:,} exploratory draws, '
          f'{agent.committed:,} committed, mean drawn duration '
          f'{agent.repeat_mean:.2f}, longest {agent.longest_draw} '
          f'(cap {agent._cap}), {agent.committed_steps:,} steps '
          f'({pct:.1f}% of the run) inside a commitment', flush=True)


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    # wrap stage 02's agent class so the summary prints without touching
    # stage 02's train() or report()
    holder = {}

    class _Instrumented(AGENT_CLS):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder['agent'] = self

    rc = stage_02_main(argv, agent_cls=_Instrumented, stage_id=STAGE_ID,
                       extra_args=extra_args, build=build)
    agent = holder.get('agent')
    if agent is not None:
        import argparse as _ap
        steps = parse_args_steps(argv)
        report_ez(agent, steps)
    return rc


def parse_args_steps(argv):
    """--total-steps without re-running the whole parser's side effects."""
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--total-steps' and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith('--total-steps='):
            return int(a.split('=', 1)[1])
    return 1_000_000


if __name__ == '__main__':
    sys.exit(main() or 0)
