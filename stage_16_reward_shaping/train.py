"""Stage 16 -- reward shaping on inventory firsts (docs/beyond-c1-c2-c3.md §3, #4).

Between holding a wood pickaxe and standing on stone there are ~30-60 steps
with no reward, so nothing tells the agent that walking toward the mountain
was worth it. This adds a bonus the first time each target item appears:

    --shape-items stone,coal,iron,diamond  --shape-bonus 0.1

Training uses the shaped reward; the reported score does not. stats.jsonl and
info['reward'] are untouched, so the number stays comparable to other runs.

Two guards against farming the bonus: it pays only on the first increase of
each item per episode, and --shape-decay-steps takes it linearly to 0 over
the last N steps so the measured policy optimises the true reward. Run
--shape-bonus 0 in the same batch as the control.

This is not potential-based shaping in Ng, Harada & Russell's sense -- a
one-off bonus on "already paid" episode history is not a difference of
potentials over a Markov state, so the policy-invariance result does not
apply and the optimum can shift. The decay is what brings it back.

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

# The four items past the wall. Wood and sapling are left out: the champion
# already gets them, so paying for them only adds variance.
DEFAULT_SHAPE_ITEMS = ('stone', 'coal', 'iron', 'diamond')


class ShapedDoubleDQN(DoubleDQN):
    """Stage 03, with a first-per-episode inventory bonus added to reward.

    Only the number entering the replay buffer changes; stats.jsonl is
    written by the Recorder inside the env, so the reported score is true.
    """

    def __init__(self, *args, shape_items=DEFAULT_SHAPE_ITEMS,
                 shape_bonus=0.1, shape_decay_steps=0, total_steps=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_items = tuple(shape_items)
        self._shape_bonus = shape_bonus
        self._shape_decay_steps = shape_decay_steps
        self._total_steps = total_steps
        # One set per env: a shared set would let env 0's first stone
        # silence env 1's.
        self._paid = [set() for _ in range(self._num_envs)]
        self._prev_inv = [None] * self._num_envs
        # For the log line: how much bonus was actually paid.
        self.shaped_total = 0.0
        self.shaped_events = 0
        # Env steps so far, for the decay; train() below keeps it current.
        self._env_steps = 0

    def _bonus_now(self, env_steps):
        """The bonus now. Linear decay to 0 over the last decay_steps."""
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
            # Strictly an increase: `> 0` would re-pay after every spend.
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

        info is supplied by this stage's own train(). It defaults to None, so
        a loop that forgets it degrades to the unshaped baseline.
        """
        reward = self.shape(i, reward, info, self._env_steps)
        return super().learn(i, state, action, reward, state_, mask, legal_)

    def end_episode(self, i):
        """Without the clear, the bonus pays once per run, not per episode."""
        super().end_episode(i)
        self._paid[i].clear()
        self._prev_inv[i] = None


def train(agent, env, step_num=100_000, log_every=10_000,
          use_mask=False, mask_extended=False, mask_skip_grass=False,
          ckpt_path=None, ckpt_every=0, start_step=0, start_episodes=0):
    """Stage 02's loop, plus _env_steps and info passed down to learn().

    Copied rather than hooked into stage 02's train(), which every other
    stage shares.
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

        # Before the batch is learned from, not after: otherwise the run's
        # last batch is shaped with a bonus the schedule has already zeroed.
        agent._env_steps = step

        for i in range(num_envs):
            next_state = states_[i]
            if dones[i] and 'final_obs' in infos[i]:
                next_state = infos[i]['final_obs']
            # info goes in so shape() sees the inventory. On a done step
            # infos[i] is the dying step's, which is the right one here.
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
    """Stage 03's net and buffer: the change is the reward, not the net."""
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
