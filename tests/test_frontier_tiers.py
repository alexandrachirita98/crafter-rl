"""The frontier must track tech-tree depth, not survival luck.

wake_up used to sit in tier 5 beside collect_diamond, and a tier counted as
reached if ANY member did. A sleeping Crafter player wakes up on its own when
energy refills, and `sleep` is one of 17 random buttons, so wake_up is
unlocked in 73-80% of episodes by every agent in the repo including the
random one. Tier 5 therefore cleared the 20% threshold at the tracker's first
decision and the monotonicity rule froze n at 3 for the rest of the run:
measured from frontier.jsonl, the only tiers ever recorded across 21,845
episodes were 1 and 5. Both 1M arms ran fixed n = 3.

See docs/performance-issue-survey.md section 3.
"""

import numpy as np

from common.metrics import ACHIEVEMENTS
from stage_13_c2_frontier_nstep.train import (SURVIVAL, TIERS, TIER_N,
                                              FrontierTracker)


def _episode(*names):
    return {name: (1 if name in names else 0) for name in ACHIEVEMENTS}


def _settle(tracker, episode, window):
    for _ in range(window):
        tracker.end_of_episode(episode)
    return tracker


def test_tiers_and_survival_cover_all_22_exactly_once():
    flat = [n for names in TIERS.values() for n in names] + list(SURVIVAL)
    assert sorted(flat) == sorted(ACHIEVEMENTS)
    assert len(flat) == len(set(flat)), 'an achievement is in two buckets'


def test_wake_up_cannot_move_the_frontier():
    """The regression, named directly."""
    assert 'wake_up' in SURVIVAL
    for tier, names in TIERS.items():
        assert 'wake_up' not in names, f'wake_up is back in tier {tier}'


def test_the_real_1m_episode_stream_stays_at_tier_1():
    """What the 1M runs actually produced: survival plus wood, no table."""
    t = _settle(FrontierTracker(window=10, threshold=0.20),
                _episode('wake_up', 'collect_sapling', 'place_plant',
                         'collect_wood', 'collect_drink'), 10)
    assert t.frontier == 1
    assert t.n == TIER_N[1]


def test_survival_only_agent_stays_at_tier_1():
    t = _settle(FrontierTracker(window=10, threshold=0.20),
                _episode(*SURVIVAL), 10)
    assert t.frontier == 1


def test_tiers_must_clear_in_order():
    """A deep unlock with nothing beneath it must not jump the ladder."""
    t = _settle(FrontierTracker(window=10, threshold=0.20),
                _episode('collect_wood', 'collect_coal'), 10)
    assert t.frontier == 1, (
        'collect_coal is tier 4 but tiers 2 and 3 are at 0%; the frontier '
        'must mean "everything shallower is also reliable"')


def test_a_real_tech_tree_climb_advances_the_frontier():
    """The mechanism still has to work, not just refuse to fire."""
    t = _settle(FrontierTracker(window=10, threshold=0.20),
                _episode('collect_wood', 'place_table', 'make_wood_pickaxe',
                         'collect_stone', 'make_stone_pickaxe'), 10)
    assert t.frontier == 3
    assert t.n == TIER_N[3]


def test_below_threshold_does_not_count():
    t = FrontierTracker(window=10, threshold=0.20)
    for i in range(10):
        names = ['collect_wood', 'place_table']
        if i == 0:
            names.append('collect_stone')     # tier 3 once: 10% < 20%
        t.end_of_episode(_episode(*names))
    assert t.frontier == 2


def test_n_never_grows_back():
    t = _settle(FrontierTracker(window=10, threshold=0.20),
                _episode('collect_wood', 'place_table', 'collect_stone'), 10)
    sharp = t.n
    _settle(t, _episode('collect_wood'), 10)
    assert t.frontier == 1
    assert t.n == sharp
