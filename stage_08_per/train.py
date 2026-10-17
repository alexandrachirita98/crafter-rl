"""Stage 08 -- prioritised replay with a sum-tree.

Schaul et al. 2016, https://arxiv.org/abs/1511.05952, Appendix B.2.1 for the
sum-tree.

Sample with probability proportional to (loss + eps) ** alpha and correct the
bias with importance weights, beta annealed 0.4 -> 1.0 over the run.

EXPECT NOTHING FROM THIS, AND THAT IS FINE. EDE deleted prioritised replay and
got 2.7x better. We build it for one reason -- contribution 3 needs the
machinery, because demo transitions get
a permanent priority bonus. A null result here is the expected outcome, not a
bug to hunt. If 200k x 2 seeds shows nothing, stage 09 inherits from
MunchausenQRDQN and uses stage 06's build, the sum-tree stays on disk for
stage 12, and the slide says "null, as predicted".

The lab's deque has no stable indices, so the prioritised buffer is a
fixed-size list with a write pointer plus SumTree from common/replay.py
(already tested by `python -m common.replay`). sample() returns the six
stage-04 tensors plus the leaf indices and the IS weights, and _update takes
those two extra arguments.

Run:
    python stage_08_per/train.py --test-per
    python stage_08_per/train.py --total-steps 2000 --warmup-steps 500 \
        --buffer-size 5000
    python stage_08_per/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.replay import SumTree, beta_by_step
from stage_02_dqn.train import DEVICE, _to_tensor, main as stage_02_main
from stage_06_qrdqn.train import build as build_06
from stage_07_munchausen.train import (MunchausenQRDQN,
                                       extra_args as extra_07)

STAGE_ID = 'stage_08_per'


class PrioritizedReplayMemory:
    """Ring buffer of six-tuples plus a SumTree over (loss + eps) ** alpha."""

    def __init__(self, size, batch_size, alpha=0.5, eps=1e-6):
        self._buffer = [None] * size
        self._size, self._ptr, self._len = size, 0, 0
        self._batch_size = batch_size
        self._tree = SumTree(size)
        self._alpha, self._eps = alpha, eps
        self.beta = 0.4                 # the agent moves this every update

    def push(self, transition):
        self._buffer[self._ptr] = transition
        # A new transition gets the largest priority seen, so it is trained on
        # at least once before its own loss has any say.
        self._tree.update(self._ptr, max(self._tree.max_leaf(), 1.0))
        self._ptr = (self._ptr + 1) % self._size
        self._len = min(self._len + 1, self._size)

    def sample(self):
        B, total = self._batch_size, self._tree.total()
        # stratified: one draw from each equal slice of the total mass
        prefix = (np.arange(B) + np.random.rand(B)) * total / B
        idx = self._tree.find(prefix)
        p = self._tree.nodes[self._tree.size + idx] / total
        w = (self._len * p) ** -self.beta
        w = w / w.max()                 # Schaul 3.4: normalise by the max
        s, a, g, s_, m, n = zip(*[self._buffer[i] for i in idx])
        col = lambda v, dt: torch.tensor(
            np.asarray(v), dtype=dt).unsqueeze(1).to(DEVICE)
        return (_to_tensor(np.stack(s)), col(a, torch.int64),
                col(g, torch.float32), _to_tensor(np.stack(s_)),
                col(m, torch.float32), col(n, torch.float32),
                idx, col(w, torch.float32))

    def update_priorities(self, idx, losses):
        self._tree.update(idx, (np.asarray(losses) + self._eps) ** self._alpha)

    def __len__(self):
        return self._len


class PERQRDQN(MunchausenQRDQN):

    def __init__(self, *args, total_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_steps = total_steps       # beta anneals over the whole run

    def _update(self, states, actions, rewards, states_, mask, n, idx, weights):
        self._buffer.beta = beta_by_step(self._step_cnt, self._total_steps)
        pred, per_sample = self._per_sample(states, actions, rewards,
                                            states_, mask, n)
        loss = (weights.squeeze(1) * per_sample).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # the per-sample quantile loss is the priority, as in Rainbow
        self._buffer.update_priorities(idx, per_sample.detach().cpu().numpy())
        self.loss = loss.item()
        self.mean_q = pred.mean().item()


def extra_args(p):
    extra_07(p)
    p.add_argument('--per-alpha', type=float, default=0.5,
                   help='priority exponent; 0 is uniform sampling')
    p.add_argument('--test-per', action='store_true',
                   help='check the tree and the beta anneal, then exit')


def build(args, capacity):
    net, _, extra = build_06(args, capacity)
    buffer = PrioritizedReplayMemory(size=capacity, batch_size=args.batch_size,
                                     alpha=args.per_alpha)
    return net, buffer, dict(extra, total_steps=args.total_steps)


# ------------------------------------------------------------- self-test


def test_per(steps=40, warmup=20, capacity=64, batch=8):
    """Run random frames through the real agent and inspect the tree.

    Every written leaf must carry a priority strictly above 0, every unwritten
    leaf must be exactly 0 (the SumTree pads its capacity to a power of two
    and those padding leaves must stay unreachable), and beta must have moved
    off its 0.4 start.
    """
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import (N_ACTIONS, get_epsilon_schedule, reset_rng)

    reset_rng(0)
    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              n_step=3, n_quantiles=11, per_alpha=0.5,
                              total_steps=steps)
    net, buffer, extra = build(args, capacity)
    agent = PERQRDQN(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                     get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                     gamma=0.99, warmup_steps=warmup, update_steps=2,
                     update_target_steps=10, **extra)

    rng = np.random.RandomState(0)
    for t in range(steps):
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.learn(s, t % N_ACTIONS, float(t % 2), s_, 1.0)

    tree = buffer._tree
    leaves = tree.nodes[tree.size:]
    written = leaves[:len(buffer)]
    unwritten = leaves[len(buffer):]

    pushed_ok = len(buffer) > 0
    pos = bool((written > 0).all())
    zero = bool((unwritten == 0).all())
    moved = buffer.beta > 0.4
    print(f'transitions stored      {len(buffer)}  (n-step window holds back '
          f'{steps - len(buffer)})')
    print(f'written leaves > 0      {pos}   want True   '
          f'(range {written.min():.2f}-{written.max():.2f})')
    print(f'unwritten leaves == 0   {zero}   want True   '
          f'({len(unwritten)} padding leaves to capacity {tree.size})')
    print(f'beta moved off 0.4      {buffer.beta:.2f}  want > 0.4')
    ok = pushed_ok and pos and zero and moved
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-per' in (argv if argv is not None else sys.argv[1:]):
        return test_per()
    return stage_02_main(argv, agent_cls=PERQRDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
