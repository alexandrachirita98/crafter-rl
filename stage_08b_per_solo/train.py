"""Stage 08b -- prioritised replay alone, on stage 03.

Stage 08 prioritises by the per-sample quantile loss, which needs stage 06;
this prioritises by |TD error| on stage 03's scalar target, Schaul's original
formulation. Same PrioritizedReplayMemory, holding five-tuples instead of six,
and _update is stage 03's loss made per-sample so it can be both weighted and
used as a priority.

Expect nothing here either. The isolated row exists so a null result lands on
PER itself rather than on PER-on-top-of-five-other-things.

Run:
    python stage_08b_per_solo/train.py --test-solo
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.replay import beta_by_step
from stage_02_dqn.train import (DEVICE, N_ACTIONS, _apply_mask, _to_tensor,
                                get_estimator, main as stage_02_main)
from stage_03_double.train import DoubleDQN
from stage_08_per.train import PrioritizedReplayMemory

STAGE_ID = 'stage_08b_per_solo'


class OneStepPrioritizedReplayMemory(PrioritizedReplayMemory):
    """Stage 08's sum-tree buffer over five-tuples instead of six."""

    def sample(self):
        B, total = self._batch_size, self._tree.total()
        prefix = (np.arange(B) + np.random.rand(B)) * total / B
        idx = self._tree.find(prefix)
        p = self._tree.nodes[self._tree.size + idx] / total
        w = (self._len * p) ** -self.beta
        w = w / w.max()                 # Schaul 3.4: normalise by the max
        s, a, r, s_, m, lg = zip(*[self._buffer[i] for i in idx])
        col = lambda v, dt: torch.tensor(
            np.asarray(v), dtype=dt).unsqueeze(1).to(DEVICE)
        return (_to_tensor(np.stack(s)), col(a, torch.int64),
                col(r, torch.float32), _to_tensor(np.stack(s_)),
                col(m, torch.float32),
                torch.from_numpy(np.stack(lg)).to(DEVICE),
                idx, col(w, torch.float32))


class PERDoubleDQN(DoubleDQN):
    """Stage 03, sampling by |TD error| ** alpha with IS correction."""

    def __init__(self, *args, total_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_steps = total_steps       # beta anneals over the whole run

    def _update(self, states, actions, rewards, states_, mask, legal_,
                idx, weights):
        self._buffer.beta = beta_by_step(self._step_cnt, self._total_steps)
        with torch.no_grad():
            actions_ = _apply_mask(self._estimator(states_),
                                   legal_).argmax(1, keepdim=True)
            qsa_ = self._target_estimator(states_).gather(1, actions_)
        qsa = self._estimator(states).gather(1, actions)
        target_qsa = rewards + self._gamma * qsa_ * mask

        # per-sample, so the same number can be weighted AND used as priority
        td = qsa - target_qsa
        if self._loss_kind == 'mse':
            per_sample = td.pow(2)
        else:
            per_sample = nn.functional.smooth_l1_loss(
                qsa, target_qsa, reduction='none')
        loss = (weights * per_sample).mean()
        self._backward(loss)

        self._buffer.update_priorities(
            idx, td.detach().abs().squeeze(1).cpu().numpy())
        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


AGENT_CLS = PERDoubleDQN


def extra_args(p):
    p.add_argument('--per-alpha', type=float, default=0.5,
                   help='priority exponent; 0 is uniform sampling')
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation and the tree, then exit')


def build(args, capacity):
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = OneStepPrioritizedReplayMemory(
        size=capacity, batch_size=args.batch_size, alpha=args.per_alpha)
    return net, buffer, {'total_steps': args.total_steps}


# ------------------------------------------------------------- self-test


def test_solo(steps=40, warmup=20, capacity=64, batch=8):
    """Isolation, plus stage 08's tree checks on the five-tuple buffer."""
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import get_epsilon_schedule, reset_rng
    from stage_04_nstep.train import NStepDoubleDQN
    from stage_06_qrdqn.train import QRDQN

    no_nstep = not issubclass(AGENT_CLS, NStepDoubleDQN)
    no_quant = not issubclass(AGENT_CLS, QRDQN)
    print(f'agent carries no n-step     {no_nstep}   want True')
    print(f'agent carries no quantiles  {no_quant}   want True')

    reset_rng(0)
    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              per_alpha=0.5, total_steps=steps)
    net, buffer, extra = build(args, capacity)
    agent = AGENT_CLS(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                      get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                      gamma=0.99, warmup_steps=warmup, update_steps=2,
                      update_target_steps=10, **extra)

    rng = np.random.RandomState(0)
    for t in range(steps):
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.learn(0, s, t % N_ACTIONS, float(t % 2), s_, 1.0)

    tree = buffer._tree
    leaves = tree.nodes[tree.size:]
    written, unwritten = leaves[:len(buffer)], leaves[len(buffer):]
    stored = len(buffer) == steps          # one-step: every step is stored
    pos = bool((written > 0).all())
    zero = bool((unwritten == 0).all())
    moved = buffer.beta > 0.4
    print(f'\ntransitions stored          {len(buffer)}/{steps}  '
          f'{"ok" if stored else "WRONG"}  (one-step holds nothing back)')
    print(f'written leaves > 0          {pos}   want True   '
          f'(range {written.min():.2f}-{written.max():.2f})')
    print(f'unwritten leaves == 0       {zero}   want True   '
          f'({len(unwritten)} padding leaves)')
    print(f'beta moved off 0.4          {buffer.beta:.2f}  want > 0.4')

    ok = no_nstep and no_quant and stored and pos and zero and moved
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    return stage_02_main(argv, agent_cls=AGENT_CLS, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
