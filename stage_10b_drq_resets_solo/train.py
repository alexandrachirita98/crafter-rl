"""Stage 10b -- DrQ augmentation and periodic resets alone, on stage 03.

Both mechanisms are regularisers that need nothing above Double DQN: the shift
augments whatever images the buffer hands back, the reset re-initialises
whatever head the net has.

Stage 03's net is get_estimator's plain nn.Sequential, not the dueling
two-stream net, so _reset_head re-initialises that Sequential's Linear layers
instead of the value/adv streams. It reports how many layers it touched and
how many frames the buffer kept.

Run:
    python stage_10b_drq_resets_solo/train.py --test-solo
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (N_ACTIONS, ReplayMemory, get_estimator,
                                main as stage_02_main)
from stage_03_double.train import DoubleDQN
from stage_10_drq_resets.train import random_shift

STAGE_ID = 'stage_10b_drq_resets_solo'


class DrQResetsDoubleDQN(DoubleDQN):
    """Stage 03 with shifted inputs and a periodically re-initialised head."""

    def __init__(self, *args, reset_every=100_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_every = reset_every

    def _update(self, states, actions, rewards, states_, mask):
        # independent shifts for the online input and the target input
        return super()._update(random_shift(states), actions, rewards,
                               random_shift(states_), mask)

    def learn(self, *transition):
        super().learn(*transition)
        if (self._step_cnt > self._warmup_steps
                and self._step_cnt % self._reset_every == 0):
            self._reset_head()

    def _reset_head(self):
        """Re-init the Linear layers. The conv trunk and the buffer stay.

        Stage 03's estimator is one Sequential, so "the head" is its Linear
        layers; the cumulative stage 10 resets the value/adv streams instead.
        """
        n = 0
        for layer in self._estimator:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
                n += 1
        self._target_estimator.load_state_dict(self._estimator.state_dict())
        self._optimizer.state.clear()   # Adam's moments belong to the old head
        print(f'step {self._step_cnt:>8,}  RESET: re-initialised {n} head '
              f'layer(s), buffer kept ({len(self._buffer):,} frames)',
              flush=True)
        return n


AGENT_CLS = DrQResetsDoubleDQN


def extra_args(p):
    p.add_argument('--reset-every', type=int, default=100_000)
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation, the shift and one reset, then exit')


def build(args, capacity):
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {'reset_every': args.reset_every}


# ------------------------------------------------------------- self-test


def test_solo(steps=40, warmup=20, reset_every=30, capacity=64, batch=8):
    """Isolation, the shift, and one reset that keeps the buffer."""
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
    x = torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8)
    y = random_shift(x)
    shift_ok = y.shape == x.shape and y.dtype == torch.uint8
    moved = not torch.equal(x, y)
    print(f'\nshift keeps {tuple(y.shape)} uint8  {shift_ok}   want True')
    print(f'shift actually moves pixels        {moved}   want True')

    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              reset_every=reset_every)
    net, buffer, extra = build(args, capacity)
    agent = AGENT_CLS(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                      get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                      gamma=0.99, warmup_steps=warmup, update_steps=2,
                      update_target_steps=10 ** 9, **extra)

    rng = np.random.RandomState(0)
    print(f'\nrunning {steps} steps with reset_every={reset_every}:')
    head_changed = synced = False
    buf_at_reset = 0
    # Detect the reset from _reset_head itself rather than computing which
    # call it lands on: learn() increments _step_cnt before the reset check,
    # so the resetting call is the one entering at reset_every - 1, and one
    # call later the head has already moved again.
    fired = []
    real_reset = agent._reset_head

    def spy():
        n = real_reset()
        fired.append(n)
        return n

    agent._reset_head = spy
    for t in range(steps):
        pre = net[-1].weight.detach().clone()
        before = len(fired)
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.learn(0, s, t % N_ACTIONS, float(t % 2), s_, 1.0)
        if len(fired) > before and not head_changed:      # the call that reset
            head_changed = not torch.equal(pre, net[-1].weight.detach())
            synced = torch.equal(net[-1].weight.detach(),
                                 agent._target_estimator[-1].weight.detach())
            buf_at_reset = len(buffer)
    agent._reset_head = real_reset

    # The trunk is the convs inside net[0]; net[1] is the first Linear, which
    # _reset_head should re-initialise. Checked against _reset_head alone,
    # since the reset step also moves the convs legitimately.
    conv = [m for m in net[0] if isinstance(m, nn.Conv2d)]
    pre_trunk = [c.weight.detach().clone() for c in conv]
    n_layers = agent._reset_head()
    trunk_kept = all(torch.equal(a, c.weight.detach())
                     for a, c in zip(pre_trunk, conv))

    print(f'\nat the reset step:')
    print(f'  head re-initialised      {head_changed}   want True')
    print(f'  target net re-synced     {synced}   want True')
    print(f'  buffer kept its frames   {buf_at_reset} frames  want > 0')
    print(f'  resets fired in {steps} steps  {len(fired)}  '
          f'want {steps // reset_every}')
    print(f'_reset_head touches {n_layers} Linear layer(s), all '
          f'{len(conv)} convs untouched {trunk_kept}   want True')

    ok = (no_nstep and no_quant and shift_ok and moved and head_changed
          and synced and buf_at_reset > 0 and trunk_kept and n_layers == 2
          and len(fired) == steps // reset_every)
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    return stage_02_main(argv, agent_cls=AGENT_CLS, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
