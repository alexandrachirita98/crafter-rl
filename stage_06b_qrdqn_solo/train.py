"""Stage 06b -- QR-DQN alone, on stage 03.

Stage 06 sits on n-step returns and the dueling head; this changes only the
loss and the head width -- 51 quantiles per action, quantile Huber instead of
MSE, one-step targets, no dueling split, no n-step window.

The one-step target is why this is not a two-line subclass: the cumulative
QRDQN discounts by gamma ** n from stage 04's six-tuples, while here the
buffer hands back stage 02's five-tuples and the discount is a plain gamma.
Loss, taus and the per-sample split are imported from stage 06.

Run:
    python stage_06b_qrdqn_solo/train.py --test-solo
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.networks import quantile_midpoints
from stage_02_dqn.train import (DEVICE, N_ACTIONS, ReplayMemory, _to_tensor,
                                get_estimator, main as stage_02_main)
from stage_03_double.train import DoubleDQN
from stage_06_qrdqn.train import quantile_huber_loss

STAGE_ID = 'stage_06b_qrdqn_solo'


class QRNet(nn.Module):
    """get_estimator's trunk and a single head N quantiles wide: (B, A, N).

    No value/advantage split -- that is stage 05's change, and keeping it out
    is the entire point of this variant.
    """

    def __init__(self, action_num, n_quantiles=51, lin_size=64, stride=1):
        super().__init__()
        plain = get_estimator(action_num, lin_size=lin_size, stride=stride)
        self.trunk = plain[0]                 # ByteToFloat, three convs, View
        flat = plain[1].in_features           # the probed flatten width
        self.action_num, self.n_quantiles = action_num, n_quantiles
        self.head = nn.Sequential(
            nn.Linear(flat, lin_size), nn.ReLU(inplace=True),
            nn.Linear(lin_size, action_num * n_quantiles))

    def forward(self, x):
        f = self.trunk(x)
        return self.head(f).view(-1, self.action_num, self.n_quantiles)


class QRDoubleDQN(DoubleDQN):
    """Stage 03's agent with a quantile head and a one-step quantile target."""

    def __init__(self, *args, n_quantiles=51, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_quant = n_quantiles
        self._taus = quantile_midpoints(n_quantiles, DEVICE)

    def _q(self, states):
        """The mean over quantiles IS the Q-value. (B, A)."""
        return self._estimator(states).mean(2)

    def step(self, states):
        n = len(states)
        if self._step_cnt < self._warmup_steps:
            return np.random.randint(self._action_num, size=n)
        self.eps = next(self._epsilon)
        greedy = np.random.rand(n) >= self.eps
        actions = np.random.randint(self._action_num, size=n)
        if greedy.any():
            with torch.no_grad():
                actions[greedy] = self._q(
                    _to_tensor(states[greedy])).argmax(1).cpu().numpy()
        return actions

    def _update(self, states, actions, rewards, states_, mask):
        pred, per_sample = self._per_sample(states, actions, rewards,
                                            states_, mask)
        loss = per_sample.mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.loss = loss.item()
        self.mean_q = pred.mean().item()

    def _per_sample(self, states, actions, rewards, states_, mask):
        with torch.no_grad():
            target = self._target_quantiles(states, actions, rewards,
                                            states_, mask)
        idx = actions.unsqueeze(2).expand(-1, 1, self._n_quant)
        pred = self._estimator(states).gather(1, idx).squeeze(1)
        return pred, quantile_huber_loss(pred, target, self._taus)

    def _target_quantiles(self, states, actions, rewards, states_, mask):
        """One-step Double-DQN target, N quantiles wide. gamma, not gamma**n."""
        a_ = self._q(states_).argmax(1)                          # online picks
        idx = a_.view(-1, 1, 1).expand(-1, 1, self._n_quant)
        quant_ = self._target_estimator(states_).gather(1, idx).squeeze(1)
        return rewards + self._gamma * mask * quant_


AGENT_CLS = QRDoubleDQN


def extra_args(p):
    p.add_argument('--n-quantiles', type=int, default=51)
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation and the one-step target, then exit')


def build(args, capacity):
    net = QRNet(N_ACTIONS, n_quantiles=args.n_quantiles,
                lin_size=args.lin_size, stride=args.stride).to(DEVICE)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {'n_quantiles': args.n_quantiles}


# ------------------------------------------------------------- self-test


def test_solo(batch=8, n_quant=11, gamma=0.99):
    """Isolation plus the one piece of new code: the one-step quantile target."""
    import argparse
    from stage_04_nstep.train import NStepDoubleDQN
    from stage_05_dueling.train import DuelingNet

    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              n_quantiles=n_quant)
    net, buffer, extra = build(args, 64)

    no_nstep = not issubclass(AGENT_CLS, NStepDoubleDQN)
    plain_buf = type(buffer).__name__ == 'ReplayMemory'
    no_dueling = not isinstance(net, DuelingNet) and not hasattr(net, 'value')
    x = torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8,
                      device=DEVICE)
    with torch.no_grad():
        shape_ok = tuple(net(x).shape) == (batch, N_ACTIONS, n_quant)
    print(f'agent carries no n-step        {no_nstep}   want True')
    print(f'buffer is one-step             {plain_buf}   want True')
    print(f'net has no duelling split      {no_dueling}   want True')
    print(f'output is (B, 17, {n_quant})         {shape_ok}   want True')

    # the one-step target: with mask 0 the target is the reward, broadcast
    # over every quantile; with mask 1 it adds exactly gamma * next quantiles
    agent = AGENT_CLS.__new__(AGENT_CLS)
    agent._gamma, agent._n_quant = gamma, n_quant
    fixed = torch.randn(batch, N_ACTIONS, n_quant)
    agent._estimator = agent._target_estimator = lambda s: fixed
    actions = torch.randint(0, N_ACTIONS, (batch, 1))
    rewards = torch.randn(batch, 1)
    states = torch.zeros(batch, 3, 64, 64, dtype=torch.uint8)

    t0 = agent._target_quantiles(states, actions, rewards, states,
                                 torch.zeros(batch, 1))
    terminal = bool((t0 - rewards).abs().max() < 1e-6)
    t1 = agent._target_quantiles(states, actions, rewards, states,
                                 torch.ones(batch, 1))
    a_ = fixed.mean(2).argmax(1)
    want = rewards + gamma * fixed[torch.arange(batch), a_]
    boots = bool((t1 - want).abs().max() < 1e-5)
    print(f'mask=0 -> target is the reward {terminal}   want True')
    print(f'mask=1 -> reward + gamma*Q(s") {boots}   want True  '
          '(one-step, no gamma**n)')

    ok = no_nstep and plain_buf and no_dueling and shape_ok and terminal and boots
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    return stage_02_main(argv, agent_cls=AGENT_CLS, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
