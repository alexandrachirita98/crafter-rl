"""Stage 13 -- n driven by the tech-tree frontier.

Our own idea, and the one most likely to do nothing: if the agent never gets
past tier 2 the dial never moves and this reduces to fixed n = 10. Hence built
last, after stage 12 gives the agent something to climb with.

BBF (Schwarzer et al. 2023, https://arxiv.org/abs/2305.19452) section 3 anneals
n from 10 to 3 on the step counter. Our change is the trigger, not the
schedule: a step counter is a guess at progress, where the frontier is the
progress. Two seeds reaching the stone pickaxe at 300k and 600k want the dial
turned at different moments, and a fixed schedule has to get one of them
wrong. Long n while the agent is still finding its first rewards, because
reward has to travel far back; short n once it is deep in the tech tree, where
targets are dense and long n is mostly variance.

The tracker sets the agent's live `_n_step` (what stage 04's window reads) and
only at an episode boundary, after end_episode() has flushed -- mid-episode it
would leave the open window holding a mix of lengths. It needs
info['achievements'] at the done step, which stage 11's loop already carries.

Controls, all flags on this file:
    --fixed-n 3      BBF's endpoint
    --fixed-n 10     stage 04's default, the real baseline
    --bbf-schedule   step-based annealing, the published trigger

Run:
    python stage_13_c2_frontier_nstep/train.py --test-frontier
    python stage_13_c2_frontier_nstep/train.py --total-steps 200000 --seed 1
"""

import json
import pathlib
import sys
import time
from collections import deque

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import bootstrap_mask
from common.masking import legal_mask
from common.metrics import ACHIEVEMENTS
from stage_02_dqn.train import main as stage_02_main
from stage_11_c1_achievement_aux.train import counts_vector, pick_base
from stage_12_c3_demonstrations.train import (DQfD, DQfD03, build as build_12,
                                              build_base03 as build_12_base03,
                                              extra_args as extra_12)

STAGE_ID = 'stage_13_c2_frontier_nstep'

# Read off crafter/data.yaml and checked against common.metrics.ACHIEVEMENTS.
# The assert below earns its keep: a typo here would leave an achievement in
# no tier at all, and nothing else would complain.
TIERS = {
    1: ['collect_wood', 'collect_drink', 'collect_sapling', 'eat_cow'],
    2: ['place_table', 'make_wood_pickaxe', 'make_wood_sword', 'place_plant'],
    3: ['collect_stone', 'place_stone', 'make_stone_pickaxe',
        'make_stone_sword'],
    4: ['collect_coal', 'collect_iron', 'place_furnace',
        'make_iron_pickaxe', 'make_iron_sword'],
    5: ['collect_diamond', 'eat_plant', 'defeat_zombie',
        'defeat_skeleton', 'wake_up'],
}
TIER_N = {1: 10, 2: 8, 3: 5, 4: 3, 5: 3}

_flat = [name for names in TIERS.values() for name in names]
assert sorted(_flat) == sorted(ACHIEVEMENTS), 'TIERS does not cover the 22'


class FrontierTracker:
    """n follows achieved depth, and only ever shrinks.

    The window (100 episodes), the threshold (20%) and the monotonicity rule
    are ours and untuned. If the stage looks marginal, these three are the
    first things to ablate.
    """

    def __init__(self, window=100, threshold=0.20, tier_n=None):
        self.history = deque(maxlen=window)
        self.threshold = threshold
        self._tier_n = tier_n or TIER_N
        self.frontier = 1
        self.n = self._tier_n[1]             # start where the schedule starts

    def end_of_episode(self, achievements):
        """achievements: {name: count} exactly as info['achievements'] gives."""
        self.history.append({k: v > 0 for k, v in achievements.items()})
        if len(self.history) < self.history.maxlen:
            return self.n                    # not enough data yet
        frontier = 1
        for tier, names in sorted(TIERS.items()):
            rate = np.mean([any(ep[nm] for nm in names)
                            for ep in self.history])
            if rate >= self.threshold:       # reached in >= 20% of episodes
                frontier = tier
        self.frontier = frontier
        # n never grows back. A lucky streak should not undo the sharpening,
        # or n oscillates and training never settles.
        self.n = min(self.n, self._tier_n[frontier])
        return self.n


def bbf_n(step, total_steps, n_start=10, n_end=3, anneal_frac=0.5):
    """The published control: n annealed on the step counter, not progress."""
    frac = min(1.0, step / max(1, anneal_frac * total_steps))
    return int(round(n_start + frac * (n_end - n_start)))


class FrontierMixin:
    """_n_step under the tracker's control. Base-independent."""

    def __init__(self, *args, frontier_window=100, frontier_threshold=0.20,
                 fixed_n=None, bbf_schedule=False, tier_n=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = FrontierTracker(window=frontier_window,
                                        threshold=frontier_threshold,
                                        tier_n=tier_n)
        self._fixed_n = fixed_n
        self._bbf = bbf_schedule
        if fixed_n is not None:
            self._n_step = fixed_n
        else:
            # Without this the agent runs at --n-step's value (10 by default)
            # until the tracker's first decision, 100 episodes in -- even
            # when the schedule's tier 1 says something shorter.
            self._n_step = self._tracker.n

    def on_episode_end(self, achievements):
        """Called by the loop after end_episode() has flushed the window.

        Never mid-episode: stage 04's window would then hold a mix of lengths,
        and the n stored on each sample would stop describing it.
        """
        if self._fixed_n is not None:
            return
        if self._bbf:
            self._n_step = bbf_n(self._step_cnt, self._total_steps)
            return
        self._n_step = self._tracker.end_of_episode(achievements)


class FrontierDQfD(FrontierMixin, DQfD):
    """Stage 12 on the stage-10 base, with n driven by the frontier."""


class FrontierDQfD03(FrontierMixin, DQfD03):
    """Stage 12 on the stage-03 base, with n driven by the frontier."""


def train_frontier(agent, env, step_num=100_000, log_every=10_000,
                   use_mask=False):
    """Stage 11's loop, plus the per-episode frontier log.

    frontier.jsonl is worth keeping even if n never moves: "the dial stayed at
    10 because the agent plateaued at tier 2" is only a conclusion if there is
    a curve behind it.
    """
    logdir = pathlib.Path(getattr(env, '_directory', '.'))
    flog = (logdir / 'frontier.jsonl').open('a')

    N = 0
    ep_rewards = [0.0]
    # main() hands this stage a VecEnv(num_envs=1), not a bare crafter.Env,
    # so everything here is a length-1 batch. See stage 11's loop.
    state, done = env.reset()[0], False
    start = time.time()
    mask = None                     # reset() returns no info

    for step in range(step_num):
        action = agent.step(state[None], mask)[0]
        state_, reward, done, info = env.step([action])
        state_, reward, done, info = state_[0], reward[0], done[0], info[0]
        if use_mask:
            mask = legal_mask(info)[None]
        agent.learn(0, state, action, reward, state_,
                    bootstrap_mask(done, info),
                    counts=counts_vector(info))
        state = state_
        ep_rewards[N] += reward

        if done:
            achievements = info['achievements']
            agent.end_episode(0)
            agent.on_episode_end(achievements)
            flog.write(json.dumps({
                'step': step + 1, 'episode': N,
                'frontier': agent._tracker.frontier,
                'n_step': agent._n_step,
                # float(): VecEnv rewards are np.float32, which json
                # refuses to encode.
                'reward': float(ep_rewards[N]),
            }) + '\n')
            flog.flush()
            # VecEnv auto-resets; state_ is already the next episode's first
            # frame, so resetting again would discard it.
            done = False
            mask = None             # new world, new player: nothing owned
            ep_rewards.append(0.0)
            N += 1

        if (step + 1) % log_every == 0:
            fps = (step + 1) / (time.time() - start)
            recent = ep_rewards[-11:-1] or [float('nan')]
            print('[{0:4d}][{1:8d}], R/ep={2:6.2f}, eps={3:4.2f}, '
                  'loss={4:8.4f}, meanQ={5:7.3f}, tier={6:d}, n={7:d}, '
                  '{8:5.0f} fps'.format(
                      N, step + 1, float(np.mean(recent)), agent.eps,
                      agent.loss, agent.mean_q, agent._tracker.frontier,
                      agent._n_step, fps), flush=True)

    flog.close()
    return None, N, time.time() - start


def extra_args(p):
    extra_12(p)
    p.add_argument('--frontier-window', type=int, default=100,
                   help='episodes averaged before the frontier may move')
    p.add_argument('--frontier-threshold', type=float, default=0.20,
                   help='a tier counts as reached at this episode rate')
    p.add_argument('--fixed-n', type=int, default=None,
                   help='control arm: pin n and disable the tracker')
    p.add_argument('--bbf-schedule', action='store_true',
                   help="control arm: BBF's step-based anneal, 10 -> 3")
    p.add_argument('--tier-n', type=str, default=None,
                   help='n per tier 1..5 as a comma list, e.g. 3,3,5,3,3. '
                        'Default is the file constant, 10,8,5,3,3')
    p.add_argument('--test-frontier', action='store_true',
                   help='check the tiers, the threshold and monotonicity')


def _tier_n(args):
    if not args.tier_n:
        return None
    vals = [int(v) for v in args.tier_n.split(',')]
    assert len(vals) == 5, '--tier-n wants five values, tiers 1..5'
    return dict(zip(range(1, 6), vals))


def _frontier_extra(args, extra):
    return dict(extra, frontier_window=args.frontier_window,
                frontier_threshold=args.frontier_threshold,
                fixed_n=args.fixed_n, bbf_schedule=args.bbf_schedule,
                tier_n=_tier_n(args))


def build(args, capacity):
    net, buffer, extra = build_12(args, capacity)
    return net, buffer, _frontier_extra(args, extra)


def build_base03(args, capacity):
    net, buffer, extra = build_12_base03(args, capacity)
    return net, buffer, _frontier_extra(args, extra)


BASES = {
    'stage03': (FrontierDQfD03, build_base03),
    'stage10': (FrontierDQfD, build),
}


# ------------------------------------------------------------- self-test


def _episode(*names):
    return {name: (1 if name in names else 0) for name in ACHIEVEMENTS}


def test_frontier(window=10, threshold=0.20):
    """Scripted episode streams against hand-computed frontiers."""
    ok = True

    # Nothing may move before the window fills. A tracker that acted on
    # 3 episodes would chase noise for the whole run.
    t = FrontierTracker(window=window, threshold=threshold)
    for _ in range(window - 1):
        t.end_of_episode(_episode('collect_wood', 'place_table',
                                  'collect_stone'))
    warm = t.n == TIER_N[1] and t.frontier == 1
    print(f'cold start: n stays {t.n} for the first {window - 1} episodes  '
          f'{warm}   want True')
    ok &= warm

    # Tier 3 in every episode, tier 4 in none -> frontier 3 -> n = 5.
    t.end_of_episode(_episode('collect_wood', 'place_table', 'collect_stone'))
    deep = t.frontier == 3 and t.n == TIER_N[3]
    print(f'tier 3 in 100% of episodes: frontier {t.frontier}, n {t.n}   '
          f'want 3, {TIER_N[3]}   {"ok" if deep else "WRONG"}')
    ok &= deep

    # Below the threshold does not count: 1 in 10 is 10%, under the 20%.
    t2 = FrontierTracker(window=window, threshold=threshold)
    for i in range(window):
        names = ['collect_wood', 'place_table']
        if i == 0:
            names.append('collect_iron')          # tier 4, once only
        t2.end_of_episode(_episode(*names))
    sub = t2.frontier == 2 and t2.n == TIER_N[2]
    print(f'tier 4 in 10% of episodes: frontier {t2.frontier}, n {t2.n}   '
          f'want 2, {TIER_N[2]} (10% < 20% threshold)   '
          f'{"ok" if sub else "WRONG"}')
    ok &= sub

    # Monotonicity: once sharpened, a regression must not lengthen n again.
    t3 = FrontierTracker(window=window, threshold=threshold)
    for _ in range(window):
        t3.end_of_episode(_episode('collect_wood', 'place_table',
                                   'collect_stone', 'collect_coal'))
    sharp_n, sharp_f = t3.n, t3.frontier
    for _ in range(window):
        t3.end_of_episode(_episode('collect_wood'))   # collapse back to tier 1
    mono = t3.n == sharp_n and t3.frontier == 1
    print(f'after collapse to tier 1: frontier {t3.frontier}, n {t3.n}   '
          f'want 1, {sharp_n} (n must NOT grow back from {sharp_f})   '
          f'{"ok" if mono else "WRONG"}')
    ok &= mono

    # A custom schedule must actually be used. A tracker that fell back to
    # TIER_N would look identical in the log line and in frontier.jsonl.
    custom = {1: 3, 2: 3, 3: 5, 4: 3, 5: 3}
    t4 = FrontierTracker(window=window, threshold=threshold, tier_n=custom)
    starts_short = t4.n == 3
    for _ in range(window):
        t4.end_of_episode(_episode('collect_wood', 'place_table',
                                   'collect_stone'))
    # frontier 3 says 5, but n never grows, so it stays at 3.
    held = t4.frontier == 3 and t4.n == 3
    print(f'\n--tier-n 3,3,5,3,3: starts at n {t4.n}   {starts_short}   '
          f'want True')
    print(f'  frontier {t4.frontier} wants n {custom[3]}, n stays {t4.n}   '
          f'{held}   want True (monotone)')
    ok &= starts_short and held

    # The BBF control, for the ablation table.
    print(f'\nBBF step-based control, 1M budget')
    bbf = [(s, bbf_n(s, 1_000_000)) for s in (0, 250_000, 500_000, 1_000_000)]
    for s, n in bbf:
        print(f'  step {s:>9,}  n = {n}')
    bbf_ok = bbf[0][1] == 10 and bbf[-1][1] == 3
    print(f'  anneals 10 -> 3   {bbf_ok}   want True')
    ok &= bbf_ok

    # Worth printing because it is the most likely outcome.
    print(f'\nIf the frontier never leaves tier 2, n stays {TIER_N[2]} and '
          f'this stage is fixed n.\nThat is a result, not a bug -- '
          f'frontier.jsonl is the evidence either way.')
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-frontier' in (argv if argv is not None else sys.argv[1:]):
        return test_frontier()
    if '--num-envs' in (argv if argv is not None else sys.argv[1:]):
        # These stages replace stage 02's loop with a single-env one, so a
        # parallel request must fail loudly rather than be ignored.
        import argparse as _ap
        _p = _ap.ArgumentParser(add_help=False)
        _p.add_argument('--num-envs', type=int, default=1)
        if _p.parse_known_args(argv if argv is not None else sys.argv[1:])[0].num_envs > 1:
            raise SystemExit(
                f'{STAGE_ID} runs a single-env loop (its labeller needs one '
                'continuous trajectory); --num-envs > 1 is not supported')

    import stage_02_dqn.train as s02
    original = s02.train
    s02.train = train_frontier
    agent_cls, build_fn = BASES[pick_base(argv)]
    try:
        return stage_02_main(argv, agent_cls=agent_cls, stage_id=STAGE_ID,
                             extra_args=extra_args, build=build_fn)
    finally:
        s02.train = original


if __name__ == '__main__':
    sys.exit(main() or 0)
