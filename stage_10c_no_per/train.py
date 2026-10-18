"""Stage 10c -- the full stack minus prioritised replay: 04+05+06+07+09+10.

Ablation of stage_10_drq_resets -- same six mechanisms, uniform replay instead
of the sum-tree:

    04 n-step returns        05 dueling head      06 QR-DQN
    07 Munchausen            09 epsilon-z-greedy  10 DrQ + head resets

This is the contingency stage 08 described. It is the cheaper agent -- no
sum-tree updates, no priority bookkeeping -- so if it matches stage 10 within
noise, this is the one to recommend and PER waits for stage 12, which needs
the machinery for the demonstration bonus rather than for any gain of its own.

No new method here. EzGreedy and DrQResets reach into stage 08 only through
_per_sample and _update, which QRDQN also provides at the six-argument
signature, so both behaviours re-mix onto MunchausenQRDQN unchanged. What goes
away is beta: no importance weights, nothing to anneal, so --total-steps is no
longer passed into the agent.

Run:
    python stage_10c_no_per/train.py --test-no-per
    python stage_10c_no_per/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import _to_tensor, main as stage_02_main
from stage_06_qrdqn.train import build as build_06, extra_args as extra_06
from stage_07_munchausen.train import MunchausenQRDQN
from stage_10_drq_resets.train import random_shift

STAGE_ID = 'stage_10c_no_per'


class NoPerStack(MunchausenQRDQN):
    """Stages 09 and 10's behaviour on MunchausenQRDQN instead of PERQRDQN.

    Subclassing DrQResets and EzGreedy does not work: both inherit PERQRDQN,
    so it stays in any MRO built from them and brings its eight-argument
    _update along. Borrowing the methods as unbound functions does not work
    either -- their zero-argument super() is compiled against the defining
    class, so it still routes into stage 08.

    So the overrides are spelled out: stage 09's step() and end_episode(),
    stage 10's _per_sample(), learn() and _reset_head(), each differing only
    in which class super() reaches. random_shift is imported as-is. Changes
    to stage 09 or 10 do not propagate here.
    """

    def __init__(self, *args, ez_mu=2.0, ez_max_repeat=100,
                 reset_every=100_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu, self._cap = ez_mu, ez_max_repeat
        self._repeat_action, self._repeat_left = 0, 0
        self._reset_every = reset_every

    # ---------------------------------------------------- stage 09

    def step(self, state):
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()
        # Advance the schedule every step, repeating or not, so epsilon
        # decays on the same clock as the other stages.
        self.eps = next(self._epsilon)
        if self._repeat_left > 0:
            self._repeat_left -= 1
            return self._repeat_action
        if torch.rand(1).item() < self.eps:
            self._repeat_action = torch.randint(self._action_num, (1,)).item()
            # The cap is not optional -- see stage 09 for the draw statistics.
            self._repeat_left = min(int(np.random.zipf(self._mu)), self._cap) - 1
            return self._repeat_action
        with torch.no_grad():
            return self._q(_to_tensor(state)).argmax().item()

    def end_episode(self):
        super().end_episode()               # stage 04 flushes its window here
        self._repeat_left = 0               # a new world; drop the walk

    # ---------------------------------------------------- stage 10

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
    extra_06(p)
    # Stage 09's and 10's own flags, minus --per-alpha. Re-declared rather
    # than borrowed because extra_09/extra_10 pull in stage 08's parser.
    p.add_argument('--ez-mu', type=float, default=2.0)
    p.add_argument('--ez-max-repeat', type=int, default=100)
    p.add_argument('--reset-every', type=int, default=100_000)
    p.add_argument('--test-no-per', action='store_true',
                   help='check the MRO, the uniform batch and a reset, then exit')


def build(args, capacity):
    # build_06 -> build_04's uniform NStepReplayMemory, no sum-tree, and no
    # total_steps in extra because there is no beta to anneal.
    net, buffer, extra = build_06(args, capacity)
    return net, buffer, dict(extra, ez_mu=args.ez_mu,
                             ez_max_repeat=args.ez_max_repeat,
                             reset_every=args.reset_every)


# ------------------------------------------------------------- self-test


def test_no_per(steps=40, warmup=20, reset_every=30, capacity=64, batch=8):
    """Check the splice, not the mechanisms -- each has its own stage test.

    Four things can break when PER is pulled out from under stages 09 and 10:

      1. the buffer is the uniform n-step one and sample() returns six
         tensors, not eight;
      2. _update resolves to QRDQN's and the target to Munchausen's, so no
         stage-08 method is left anywhere in the MRO;
      3. the committed walk still runs, which is stage 09 surviving the swap;
      4. the reset still fires and still keeps the buffer, which is stage 10
         surviving it.
    """
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import (N_ACTIONS, get_epsilon_schedule, reset_rng)
    from stage_04_nstep.train import NStepReplayMemory
    from stage_06_qrdqn.train import QRDQN
    from stage_07_munchausen.train import MunchausenQRDQN as M
    from stage_08_per.train import PERQRDQN, PrioritizedReplayMemory

    reset_rng(0)
    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              n_step=3, n_quantiles=11, ez_mu=2.0,
                              ez_max_repeat=100, reset_every=reset_every)
    net, buffer, extra = build(args, capacity)
    agent = NoPerStack(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                       get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                       gamma=0.99, warmup_steps=warmup, update_steps=2,
                       update_target_steps=10, **extra)

    # 1. the buffer
    uniform = isinstance(buffer, NStepReplayMemory)
    not_per = not isinstance(buffer, PrioritizedReplayMemory)
    print(f'buffer is the n-step uniform one   {uniform}   want True')
    print(f'buffer is not prioritised          {not_per}   want True')
    print(f'no total_steps passed to the agent '
          f'{"total_steps" not in extra}   want True  (no beta to anneal)')

    # 2. the MRO
    per_gone = PERQRDQN not in type(agent).__mro__
    upd = type(agent)._update is QRDQN._update
    tgt = type(agent)._target_quantiles is M._target_quantiles
    print(f'\nPERQRDQN out of the MRO            {per_gone}   want True')
    print(f'_update is QRDQN\'s (6 args)         {upd}   want True')
    print(f'_target_quantiles is Munchausen\'s   {tgt}   want True')

    # 3 and 4. run frames and watch both behaviours
    rng = np.random.RandomState(0)
    repeats, reset_fired = 0, False
    print(f'\nrunning {steps} steps with reset_every={reset_every}:')
    for t in range(steps):
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pre_head = net.adv[-1].weight.detach().clone()
        agent.step(s)
        if agent._repeat_left > 0:
            repeats += 1
        agent.learn(s, t % N_ACTIONS, float(t % 2), s_, 1.0)
        if agent._step_cnt == reset_every + 1:
            reset_fired = not torch.equal(pre_head,
                                          net.adv[-1].weight.detach())

    walked = repeats > 0
    kept = len(buffer) > 0
    batch_six = len(buffer.sample()) == 6
    print(f'\nez-greedy committed a walk         {walked}   want True  '
          f'({repeats} steps inside a repeat)')
    print(f'sample() returns 6 tensors         {batch_six}   want True')
    print(f'head reset fired                   {reset_fired}   want True')
    print(f'buffer kept its data               {kept}   want True  '
          f'({len(buffer)} transitions)')

    ok = (uniform and not_per and per_gone and upd and tgt and walked
          and batch_six and reset_fired and kept
          and 'total_steps' not in extra)
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-no-per' in (argv if argv is not None else sys.argv[1:]):
        return test_no_per()
    return stage_02_main(argv, agent_cls=NoPerStack, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
