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

from common.masking import random_legal
from stage_02_dqn.train import (_apply_mask, _mask_or_ones, _to_tensor,
                                main as stage_02_main)
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
        self._repeat_action = np.zeros(self._num_envs, dtype=np.int64)
        self._repeat_left = np.zeros(self._num_envs, dtype=np.int64)
        self._reset_every = reset_every

    # ---------------------------------------------------- stage 09

    def step(self, states, masks=None):
        n = len(states)
        masks = _mask_or_ones(masks, n, self._action_num)
        if self._step_cnt < self._warmup_steps:
            return random_legal(masks)
        # Advance the schedule every step, repeating or not, so epsilon
        # decays on the same clock as the other stages.
        self.eps = next(self._epsilon)

        actions = np.zeros(n, dtype=np.int64)
        need_q = np.zeros(n, dtype=bool)
        for i in range(n):
            # One walk per env, as in stage 09: a walk is "hold this action
            # for the next k steps in THIS world". A held action can outlive
            # its legality, so a walk whose action went illegal is abandoned.
            if self._repeat_left[i] > 0 and masks[i][self._repeat_action[i]] > 0:
                self._repeat_left[i] -= 1
                actions[i] = self._repeat_action[i]
            elif np.random.rand() < self.eps:
                a = int(random_legal(masks[i][None])[0])
                # The cap is not optional -- see stage 09 for the draw stats.
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
                actions[need_q] = _apply_mask(
                    q, masks[need_q]).argmax(1).cpu().numpy()
        return actions

    def end_episode(self, i):
        super().end_episode(i)               # stage 04 flushes its window here
        self._repeat_left[i] = 0            # a new world; drop the walk

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


def test_no_per(steps=40, warmup=20, reset_every=30, capacity=64, batch=8,
                n_envs=4):
    """Check the splice, not the mechanisms -- each has its own stage test.

    Five things can break when PER is pulled out from under stages 09 and 10:

      1. the buffer is the uniform n-step one and sample() returns six
         tensors, not eight;
      2. _update resolves to QRDQN's and the target to Munchausen's, so no
         stage-08 method is left anywhere in the MRO;
      3. the committed walk still runs, which is stage 09 surviving the swap;
      4. the reset still fires and still keeps the buffer, which is stage 10
         surviving it;
      5. step() returns one action per env. This ran with n_envs=1 until a
         16-env run crashed in the train loop: a single-env step() returns a
         length-1 array, and worse, one walk shared across envs means no env
         ever holds an action for its own k steps. So the driver below steps a
         real batch.
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
                       update_target_steps=10, num_envs=n_envs, **extra)

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

    # 3, 4 and 5. run frames on a real batch and watch both behaviours.
    # _step_cnt now strides by n_envs, so the reset cannot be spotted by an
    # equality on the step number, and a weight comparison would also catch
    # every ordinary gradient step. Count the calls and check the head moved
    # across the one frame that reset.
    resets = []
    real_reset_head = agent._reset_head

    def counting_reset_head():
        resets.append(agent._step_cnt)
        real_reset_head()

    agent._reset_head = counting_reset_head

    rng = np.random.RandomState(0)
    repeats, reset_fired, widths_ok = 0, False, True
    print(f'\nrunning {steps} steps with reset_every={reset_every}, '
          f'{n_envs} envs:')
    for t in range(steps):
        s = rng.randint(0, 256, (n_envs, 64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (n_envs, 64, 64, 3), dtype=np.uint8)
        pre_head = net.adv[-1].weight.detach().clone()
        before = len(resets)
        actions = agent.step(s)
        # the bug this test used to walk past: a single-env step() hands the
        # train loop one action for sixteen envs.
        widths_ok &= np.shape(actions) == (n_envs,)
        if agent._repeat_left.max() > 0:
            repeats += 1
        for i in range(n_envs):
            agent.learn(i, s[i], t % N_ACTIONS, float(t % 2), s_[i], 1.0)
        if len(resets) > before and not torch.equal(
                pre_head, net.adv[-1].weight.detach()):
            reset_fired = True

    walked = repeats > 0
    kept = len(buffer) > 0
    batch_six = len(buffer.sample()) == 6
    print(f'\nez-greedy committed a walk         {walked}   want True  '
          f'({repeats} frames inside a repeat)')
    print(f'sample() returns 6 tensors         {batch_six}   want True')
    print(f'head reset fired                   {reset_fired}   want True  '
          f'(at step {resets})')
    print(f'buffer kept its data               {kept}   want True  '
          f'({len(buffer)} transitions)')
    print(f'step() gave one action per env     {widths_ok}   want True  '
          f'({n_envs} envs)')

    ok = (uniform and not_per and per_gone and upd and tgt and walked
          and batch_six and reset_fired and kept and widths_ok
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
