"""Stage 10 -- DrQ image augmentation plus periodic head resets.

DrQ -- Kostrikov, Yarats & Fergus 2021, https://arxiv.org/abs/2004.13649. Pad
each sampled image 4 px and random-crop back: a tree is a tree wherever it
sits in the frame.

Resets -- Nikishin et al. 2022, https://arxiv.org/abs/2205.07802. Every 100k
steps re-initialise the head and keep the buffer. Early garbage experience
leaves a permanent dent (primacy bias); this un-dents it while keeping the
data. Expect a dip in the return curve after each reset that recovers within a
few thousand steps.

_per_sample gets its inputs shifted; learn gets a reset check. The shift works
on what _to_tensor produces, uint8 NCHW, and hands back uint8 because
ByteToFloat asserts on anything else.

Run:
    python stage_10_drq_resets/train.py --test-drq
    python stage_10_drq_resets/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import main as stage_02_main
from stage_09_ez_greedy.train import (EzGreedy, build as build_09,
                                      extra_args as extra_09)

STAGE_ID = 'stage_10_drq_resets'


def random_shift(x, pad=4):
    """x: uint8 (B, 3, H, W) as _to_tensor makes it. Same shape and dtype back."""
    B, C, H, W = x.shape
    xp = F.pad(x.float(), (pad,) * 4, mode='replicate')   # replicate needs float
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    out = torch.stack([xp[b, :, i[b]:i[b] + H, j[b]:j[b] + W]
                       for b in range(B)])
    return out.to(torch.uint8)      # the values are still integers; lossless


class DrQResets(EzGreedy):

    def __init__(self, *args, reset_every=100_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_every = reset_every

    def _per_sample(self, states, actions, rewards, states_, mask, n):
        # independent shifts for the online input and the target input
        return super()._per_sample(random_shift(states), actions, rewards,
                                   random_shift(states_), mask, n)

    def learn(self, *transition):
        super().learn(*transition)
        if (self._step_cnt > self._warmup_steps
                and self._step_cnt % self._reset_every == 0):
            self._reset_head()

    def _reset_head(self):
        """Re-init the V and A streams. The trunk and the buffer stay."""
        for stream in (self._estimator.value, self._estimator.adv):
            for layer in stream:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        self._target_estimator.load_state_dict(self._estimator.state_dict())
        self._optimizer.state.clear()   # Adam's moments belong to the old head
        print(f'[reset] head re-initialised at step {self._step_cnt}',
              flush=True)


def extra_args(p):
    extra_09(p)
    p.add_argument('--reset-every', type=int, default=100_000)
    p.add_argument('--test-drq', action='store_true',
                   help='check the shift and one head reset, then exit')


def build(args, capacity):
    net, buffer, extra = build_09(args, capacity)
    return net, buffer, dict(extra, reset_every=args.reset_every)


# ------------------------------------------------------------- self-test


def test_drq(steps=40, warmup=20, reset_every=30, capacity=64, batch=8):
    """Two checks: the shift is shape- and dtype-preserving, and one reset fires.

    The reset must change the head, re-sync the target net and leave the
    buffer full -- keeping the data is the entire point of the stage.

    Both the head/target check and the head-changed check are made AT the
    reset step, not at the end of the run. Ordinary gradient steps keep
    moving the online head afterwards, so by the last step head != target
    again, which is correct behaviour and not a failed sync.
    """
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import (N_ACTIONS, get_epsilon_schedule, reset_rng)

    reset_rng(0)
    x = torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8)
    y = random_shift(x)
    shape_ok = y.shape == x.shape and y.dtype == torch.uint8
    moved = not torch.equal(x, y)
    print(f'shift keeps {tuple(y.shape)} uint8   {shape_ok}   want True')
    print(f'shift actually moves pixels        {moved}   want True')

    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              n_step=3, n_quantiles=11, per_alpha=0.5,
                              total_steps=steps, ez_mu=2.0, ez_max_repeat=100,
                              reset_every=reset_every)
    net, buffer, extra = build(args, capacity)
    agent = DrQResets(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                      get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                      gamma=0.99, warmup_steps=warmup, update_steps=2,
                      update_target_steps=10, **extra)

    rng = np.random.RandomState(0)
    print(f'\nrunning {steps} steps with reset_every={reset_every}:')
    head_changed = synced = trunk_kept = False
    for t in range(steps):
        pre_head = net.adv[-1].weight.detach().clone()
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.learn(s, t % N_ACTIONS, float(t % 2), s_, 1.0)
        if agent._step_cnt == reset_every + 1:        # the step that reset
            head_changed = not torch.equal(pre_head,
                                           net.adv[-1].weight.detach())
            synced = torch.equal(
                net.adv[-1].weight.detach(),
                agent._target_estimator.adv[-1].weight.detach())

    # Checked against _reset_head alone: the reset step also takes a gradient
    # step, which moves the trunk legitimately, so measuring there would fail
    # on correct code.
    pre_trunk = net.trunk[1].weight.detach().clone()
    agent._reset_head()
    trunk_kept = torch.equal(pre_trunk, net.trunk[1].weight.detach())

    buffer_kept = len(buffer) > 0
    print(f'\nat the reset step:')
    print(f'  adv head re-initialised  {head_changed}   want True')
    print(f'  target net re-synced     {synced}   want True')
    print(f'_reset_head leaves trunk   {trunk_kept}   want True')
    print(f'buffer kept its data       {buffer_kept}   want True  '
          f'({len(buffer)} transitions, nothing dropped)')
    ok = (shape_ok and moved and head_changed and synced and trunk_kept
          and buffer_kept)
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-drq' in (argv if argv is not None else sys.argv[1:]):
        return test_drq()
    return stage_02_main(argv, agent_cls=DrQResets, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
