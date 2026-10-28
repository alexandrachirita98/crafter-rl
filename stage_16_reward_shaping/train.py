"""Stage 16 -- reward shaping on inventory firsts (docs/beyond-c1-c2-c3.md §3, #4).

THE PROBLEM, IN ONE SENTENCE. Between "I have a wood pickaxe" (the champion
gets one in 18% of episodes) and "I am standing on stone" there are ~30-60
steps with zero reward, so nothing in the learning signal ever says that
walking toward the mountain was a good idea. The champion's achievement
profile is exactly this shape: the easy five are solved (collect_wood 86%,
place_table 68%, collect_sapling 91%) and everything from collect_stone down
is <= 3.4%.

THE FIX. A small bonus the first time per episode each target item appears in
the inventory:

    --shape-items stone,coal,iron,diamond     --shape-bonus 0.1

Train on the shaped reward, REPORT the true one. Crafter's info['reward'] is
the real environment reward and stats.jsonl records achievements, neither of
which this stage touches, so the score remains comparable to every other run.

WHY IT MIGHT NOT WORK, STATED UP FRONT. Shaping invites the agent to farm the
bonus instead of playing -- collect one stone, die, repeat. Two guards, both
from the doc:

  (a) FIRST UNIT PER EPISODE ONLY. The bonus pays on the first increase of
      each item in an episode and never again, so there is nothing to farm.
      Implemented as a per-env set of already-paid items, cleared on done.
  (b) DECAY TO ZERO. --shape-decay-steps makes the bonus fall linearly to 0
      over the last N env steps, so the final policy optimises the true
      reward. Default 300k of a 1M run, per the doc.

This is potential-based reward shaping's informal cousin (Ng, Harada & Russell
1999). SAY WHICH PART OF THAT THEORY THIS SATISFIES AND WHICH IT DOES NOT: a
potential-based shaping F(s,s') = gamma*phi(s') - phi(s) provably leaves the
optimal policy unchanged. THIS IS NOT THAT. A one-off bonus on an inventory
first is not a difference of potentials over a Markov state -- "already paid"
is episode history, not state -- so the invariance theorem does NOT apply and
the optimal policy CAN shift. Guard (b) is what brings it back: with the
bonus at 0 for the last 300k steps, the policy being measured is optimising
the true reward. Report the --shape-bonus 0 control in the same batch.

WHERE IT PLUGS IN. train() reads rewards[i] out of env.step and hands it to
agent.learn(). Shaping belongs between those two, so this stage overrides
learn() rather than touching stage 02's loop -- which keeps every other
stage's reward path untouched.

Run:
    python stage_16_reward_shaping/train.py --test-shaping
    python stage_16_reward_shaping/train.py --total-steps 200000 --seed 1 \
        --num-envs 16 --mask --seed-stride 16 --shape-bonus 0.1
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS
from stage_02_dqn.train import ReplayMemory, get_estimator, main as stage_02_main
from stage_03_double.train import DoubleDQN

STAGE_ID = 'stage_16_reward_shaping'

# The four items on the far side of the wall. Everything the champion already
# gets (wood, sapling) is deliberately absent: paying for a solved
# achievement buys nothing and only adds variance.
DEFAULT_SHAPE_ITEMS = ('stone', 'coal', 'iron', 'diamond')


class ShapedDoubleDQN(DoubleDQN):
    """Stage 03, with a first-per-episode inventory bonus added to reward.

    THE BONUS NEVER REACHES THE LOGS. stats.jsonl is written by the Recorder
    inside the env and records achievements; info['reward'] is the env's own.
    This class only changes the number that goes into the replay buffer, so
    the reported score is the true one -- which is the whole point.
    """

    def __init__(self, *args, shape_items=DEFAULT_SHAPE_ITEMS,
                 shape_bonus=0.1, shape_decay_steps=0, total_steps=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_items = tuple(shape_items)
        self._shape_bonus = shape_bonus
        self._shape_decay_steps = shape_decay_steps
        self._total_steps = total_steps
        # Per env: the items already paid for in the CURRENT episode, and the
        # previous step's inventory. One set per env -- with 16 envs a single
        # shared set would let env 0's first stone silence env 1's.
        self._paid = [set() for _ in range(self._num_envs)]
        self._prev_inv = [None] * self._num_envs
        # Counters for the log line, so the deck can report how much shaped
        # reward was actually paid rather than how much was configured.
        self.shaped_total = 0.0
        self.shaped_events = 0
        # Env steps so far, for the decay. train() below keeps it current;
        # the decay is a no-op while it is 0, which is what the self-test
        # exercises directly.
        self._env_steps = 0

    def _bonus_now(self, env_steps):
        """The bonus at this point in training. Linear decay to 0 at the end.

        With shape_decay_steps = 300k on a 1M run, the bonus is full until
        700k and 0 at 1M, so the last 300k steps optimise the true reward.
        """
        if self._shape_bonus == 0.0 or self._shape_decay_steps <= 0:
            return self._shape_bonus
        if not self._total_steps:
            return self._shape_bonus
        decay_from = self._total_steps - self._shape_decay_steps
        if env_steps <= decay_from:
            return self._shape_bonus
        frac = (env_steps - decay_from) / self._shape_decay_steps
        return self._shape_bonus * max(0.0, 1.0 - frac)

    def shape(self, i, reward, info, env_steps):
        """-> the reward to learn from. Pays once per item per episode."""
        inv = (info or {}).get('inventory')
        if inv is None:
            return reward
        prev = self._prev_inv[i]
        self._prev_inv[i] = dict(inv)
        if prev is None:
            return reward          # first step of an episode: no delta yet
        bonus = self._bonus_now(env_steps)
        if bonus == 0.0:
            return reward
        extra = 0.0
        for item in self._shape_items:
            if item in self._paid[i]:
                continue
            # Strictly an INCREASE. Crafter clamps inventory at 9 and items
            # get spent on crafting, so `> 0` would re-pay after every spend.
            if inv.get(item, 0) > prev.get(item, 0):
                self._paid[i].add(item)
                extra += bonus
        if extra:
            self.shaped_total += extra
            self.shaped_events += 1
        return reward + extra

    def learn(self, i, state, action, reward, state_, mask, legal_=None,
              info=None):
        """Shape the reward, then hand it to stage 02's learn() unchanged.

        info is an extra keyword stage 02's train() does not pass, so this
        stage ships its own train() below to supply it. Defaulting it to None
        keeps the signature compatible with every other caller -- and with
        info=None shape() is a pass-through, so a mis-wired loop degrades to
        the unshaped baseline rather than crashing fifteen frames away.
        """
        reward = self.shape(i, reward, info, self._env_steps)
        return super().learn(i, state, action, reward, state_, mask, legal_)

    def end_episode(self, i):
        """New episode, new entitlement. Without this the bonus pays once per
        RUN rather than once per episode, and the signal vanishes after the
        first few thousand steps."""
        super().end_episode(i)
        self._paid[i].clear()
        self._prev_inv[i] = None


def train(agent, env, step_num=100_000, log_every=10_000,
          use_mask=False, mask_extended=False, mask_skip_grass=False,
          ckpt_path=None, ckpt_every=0, start_step=0, start_episodes=0):
    """Stage 02's loop with two lines added: keep agent._env_steps current,
    and pass info to learn() so shape() can read the inventory.

    Copied rather than parameterised because stage 02's train() is the
    reference loop every other stage shares, and threading a per-stage
    reward hook through it would change the code path of runs that have
    nothing to do with shaping.
    """
    import signal
    import time
    from collections import deque

    from common.env import bootstrap_mask
    from common.masking import legal_masks
    from stage_02_dqn.train import save_ckpt

    num_envs = env.num_envs
    ep_reward = np.zeros(num_envs, dtype=np.float64)
    recent = deque(maxlen=10)
    states = env.reset()
    episodes = start_episodes
    start = time.time()

    dying = {'now': False}

    def _on_term(signum, frame):
        del signum, frame
        dying['now'] = True

    if ckpt_path is not None:
        signal.signal(signal.SIGTERM, _on_term)
    masks = None
    done_steps = start_step

    for step in range(start_step, step_num, num_envs):
        actions = agent.step(states, masks)
        assert np.shape(actions) == (num_envs,), (
            f'{type(agent).__name__}.step() returned {np.shape(actions)}, '
            f'want ({num_envs},) -- a single-env step() on a vec env')

        states_, rewards, dones, infos = env.step(actions)

        if use_mask:
            masks = legal_masks(infos, dones, extended=mask_extended,
                                skip_grass=mask_skip_grass)

        # The decay reads this, so it must be the step count BEFORE the
        # batch is learned from -- not after, or the last batch of the run
        # is shaped with a bonus the schedule says is already zero.
        agent._env_steps = step

        for i in range(num_envs):
            next_state = states_[i]
            if dones[i] and 'final_obs' in infos[i]:
                next_state = infos[i]['final_obs']
            # THE ONE CHANGED LINE: info goes in, so shape() can see the
            # inventory. On a done step infos[i] is the DYING step's info,
            # which is the right one -- that is the inventory the last
            # transition ended with.
            agent.learn(i, states[i], actions[i], rewards[i], next_state,
                        bootstrap_mask(dones[i], infos[i]),
                        masks[i] if use_mask else None,
                        info=infos[i])
            ep_reward[i] += rewards[i]
            if dones[i]:
                agent.end_episode(i)
                recent.append(ep_reward[i])
                ep_reward[i] = 0.0
                episodes += 1

        states = states_
        done_steps = step + num_envs

        if done_steps % log_every < num_envs:
            fps = done_steps / (time.time() - start)
            seen = list(recent) or [float('nan')]
            print('[{0:4d}][{1:8d}], R/ep={2:6.2f}, eps={3:4.2f}, '
                  'loss={4:8.4f}, meanQ={5:7.3f}, {6:5.0f} fps, '
                  'shaped={7:.1f} in {8} events, bonus={9:.4f}'.format(
                      episodes, done_steps,
                      float(np.mean(seen)), agent.eps,
                      agent.loss, agent.mean_q, fps,
                      agent.shaped_total, agent.shaped_events,
                      agent._bonus_now(done_steps)), flush=True)

        if ckpt_path is not None and ckpt_every \
                and done_steps % ckpt_every < num_envs:
            save_ckpt(ckpt_path, agent, done_steps, episodes)

        if dying['now']:
            if ckpt_path is not None:
                save_ckpt(ckpt_path, agent, done_steps, episodes)
            print(f'\nSIGTERM at {done_steps:,} steps -- checkpointed',
                  flush=True)
            break

    return done_steps, episodes, time.time() - start


def extra_args(p):
    p.add_argument('--shape-bonus', type=float, default=0.1,
                   help='added the first time each item appears in an '
                        'episode. 0 is the control arm')
    p.add_argument('--shape-items', type=str,
                   default=','.join(DEFAULT_SHAPE_ITEMS),
                   help='comma-separated inventory keys to pay for')
    p.add_argument('--shape-decay-steps', type=int, default=300_000,
                   help='decay the bonus linearly to 0 over the last N env '
                        'steps, so the final policy optimises true reward. '
                        '0 keeps it constant')
    p.add_argument('--test-shaping', action='store_true',
                   help='check the guards and the decay, then exit')


def build(args, capacity):
    """Stage 03's net and buffer: the contribution is the reward, not the net."""
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    items = tuple(s.strip() for s in args.shape_items.split(',') if s.strip())
    return net, buffer, dict(shape_items=items,
                             shape_bonus=args.shape_bonus,
                             shape_decay_steps=args.shape_decay_steps,
                             total_steps=args.total_steps)


# ------------------------------------------------------------- self-test


def _stub(num_envs=2, bonus=0.1, decay=0, total=0,
          items=DEFAULT_SHAPE_ITEMS):
    """A ShapedDoubleDQN with no net, buffer or optimizer -- shaping only."""
    a = ShapedDoubleDQN.__new__(ShapedDoubleDQN)
    a._num_envs = num_envs
    a._shape_items = tuple(items)
    a._shape_bonus = bonus
    a._shape_decay_steps = decay
    a._total_steps = total
    a._paid = [set() for _ in range(num_envs)]
    a._prev_inv = [None] * num_envs
    a.shaped_total = 0.0
    a.shaped_events = 0
    a._env_steps = 0
    return a


def _inv(**kw):
    base = {k: 0 for k in ('stone', 'coal', 'iron', 'diamond', 'wood')}
    base.update(kw)
    return {'inventory': base}


def test_shaping():
    ok = True

    # -- pays once, on an increase, and never again -------------------------
    a = _stub()
    a.shape(0, 0.0, _inv(), 0)                      # prime prev_inv
    r = a.shape(0, 0.0, _inv(stone=1), 0)
    if abs(r - 0.1) > 1e-9:
        print(f'FAIL first stone should pay 0.1, got {r}'); ok = False
    r = a.shape(0, 0.0, _inv(stone=2), 0)
    if r != 0.0:
        print(f'FAIL second stone must not pay, got {r}'); ok = False

    # Spending stone then re-collecting must NOT re-pay in the same episode.
    a.shape(0, 0.0, _inv(stone=0), 0)
    r = a.shape(0, 0.0, _inv(stone=1), 0)
    if r != 0.0:
        print(f'FAIL re-collect after spending must not pay, got {r}')
        ok = False

    # -- the true reward is preserved, not replaced -------------------------
    a = _stub()
    a.shape(0, 0.0, _inv(), 0)
    r = a.shape(0, 1.0, _inv(coal=1), 0)
    if abs(r - 1.1) > 1e-9:
        print(f'FAIL shaping must ADD to the env reward, got {r}'); ok = False

    # -- envs do not share the entitlement ---------------------------------
    a = _stub(num_envs=2)
    a.shape(0, 0.0, _inv(), 0)
    a.shape(1, 0.0, _inv(), 0)
    a.shape(0, 0.0, _inv(stone=1), 0)
    r = a.shape(1, 0.0, _inv(stone=1), 0)
    if abs(r - 0.1) > 1e-9:
        print(f'FAIL env 1 must pay independently of env 0, got {r}')
        ok = False

    # -- a new episode restores the entitlement ----------------------------
    a = _stub()
    a.shape(0, 0.0, _inv(), 0)
    a.shape(0, 0.0, _inv(stone=1), 0)
    ShapedDoubleDQN.end_episode(a, 0)
    a.shape(0, 0.0, _inv(), 0)
    r = a.shape(0, 0.0, _inv(stone=1), 0)
    if abs(r - 0.1) > 1e-9:
        print(f'FAIL a new episode should pay again, got {r}'); ok = False

    # -- decay reaches exactly zero at the end -----------------------------
    a = _stub(decay=300_000, total=1_000_000)
    for step, want in ((0, 0.1), (700_000, 0.1), (850_000, 0.05),
                       (1_000_000, 0.0)):
        got = a._bonus_now(step)
        if abs(got - want) > 1e-9:
            print(f'FAIL bonus at {step}: want {want}, got {got}'); ok = False

    # -- bonus 0 is a true no-op (the control arm) -------------------------
    a = _stub(bonus=0.0)
    a.shape(0, 0.0, _inv(), 0)
    r = a.shape(0, 2.0, _inv(stone=1, coal=1, iron=1), 0)
    if r != 2.0:
        print(f'FAIL --shape-bonus 0 must change nothing, got {r}'); ok = False
    if a.shaped_total != 0.0:
        print('FAIL control arm recorded shaped reward'); ok = False

    # -- only the configured items pay -------------------------------------
    a = _stub(items=('stone',))
    a.shape(0, 0.0, _inv(), 0)
    r = a.shape(0, 0.0, _inv(coal=1, iron=1, diamond=1), 0)
    if r != 0.0:
        print(f'FAIL unconfigured items must not pay, got {r}'); ok = False

    # -- missing inventory must not crash ----------------------------------
    a = _stub()
    if a.shape(0, 1.0, {}, 0) != 1.0 or a.shape(0, 1.0, None, 0) != 1.0:
        print('FAIL missing inventory should pass the reward through')
        ok = False

    print('reward-shaping self-test:', 'OK' if ok else 'FAILED')
    return ok


def main(argv=None):
    if '--test-shaping' in (argv if argv is not None else sys.argv[1:]):
        raise SystemExit(0 if test_shaping() else 1)
    # The loop has to pass info down to learn(), so swap stage 02's train()
    # for ours. Same mechanism stage 11 uses; restored in finally so an
    # importer of this module does not inherit the patch.
    import stage_02_dqn.train as s02
    original = s02.train
    s02.train = train
    try:
        return stage_02_main(argv, agent_cls=ShapedDoubleDQN,
                             stage_id=STAGE_ID, extra_args=extra_args,
                             build=build)
    finally:
        s02.train = original


if __name__ == '__main__':
    main()
