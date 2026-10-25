"""Demo rows must pair each state with the action taken FROM it.

The loader used to pair image[t] with action[t], but crafter's
EpisodeRecorder stores image[k] as the frame produced BY action[k] and
zero-pads index 0. So every demo row carried the action that led to its
state, the margin loss was taught "craft here" at the frame already showing
the tool, and each +1 landed one state late.

These tests run against the real demos/ files, because the bug was in the
convention and a synthetic episode would just encode whichever convention
the test author believed.

See docs/performance-issue-survey.md section 1.
"""

import pathlib

import numpy as np
import pytest

from common.metrics import ACHIEVEMENTS
from stage_12_c3_demonstrations.train import _episode_to_rows

DEMO_DIR = pathlib.Path(__file__).resolve().parents[1] / 'demos'
# place_table, place_furnace, and the four make_* actions: the presses whose
# state the margin loss has to get right.
CRAFT_ACTIONS = {8, 9, 11, 12, 14, 15, 16}


def _demo_files():
    return sorted(DEMO_DIR.glob('*.npz'))


pytestmark = pytest.mark.skipif(
    not _demo_files(), reason='no demos/*.npz; run record_demos.py first')


def test_recorder_convention_holds_in_the_files():
    """Pin the premise the loader is built on, in the data itself.

    If a future crafter version changes EpisodeRecorder, this fails first and
    names the reason, instead of the loader silently going back to being
    one step out.
    """
    hits = 0
    for f in _demo_files():
        ep = np.load(f)
        for name in ('place_table', 'make_wood_pickaxe', 'make_stone_pickaxe'):
            counts = ep[f'achievement_{name}']
            rises = np.flatnonzero(np.diff(counts) > 0)
            if not len(rises):
                continue
            k = rises[0] + 1
            # action[k] is what produced image[k], so it is the craft press
            assert ep['action'][k] in CRAFT_ACTIONS, (
                f'{f.name}: {name} rose at {k} but action[{k}] is '
                f'{ep["action"][k]}, not a craft action')
            hits += 1
        # index 0 is the zero-padded reset entry, not a transition
        assert ep['discount'][0] == 0.0
        assert ep['action'][0] == 0
    assert hits > 50, f'only {hits} craft events checked'


def test_rows_use_the_action_taken_from_the_state():
    """With n_step=1 a row is one env step, so the pairing is checkable."""
    mismatched = total = 0
    for f in _demo_files():
        ep = np.load(f)
        rows = _episode_to_rows(ep, n_step=1, gamma=0.99, horizon=10)
        actions, rewards, discounts = ep['action'], ep['reward'], ep['discount']
        want = [(int(actions[t + 1]), float(rewards[t + 1]),
                 float(discounts[t + 1])) for t in range(len(actions) - 1)]
        # row = (s0, a0, G, s_n, mask, n, legal_, y, is_demo); at n=1, G == r
        got = [(r[1], r[2], r[4]) for r in rows]
        assert len(got) == len(want)
        for (wa, wr, wd), (ga, gr, gd) in zip(want, got):
            total += 1
            if wa != ga or abs(wr - gr) > 1e-9 or abs(wd - gd) > 1e-9:
                mismatched += 1
    assert total > 20_000, f'only {total} rows checked'
    assert mismatched == 0, f'{mismatched} of {total} rows are off by one'


def test_no_fake_terminal_from_the_padded_reset_entry():
    """discount[0] is padding, so reading it produced one bogus death each.

    A terminal row says "this state has no future". Getting one on the reset
    frame of every demo teaches the opposite of what the demo is for.
    """
    for f in _demo_files():
        ep = np.load(f)
        rows = _episode_to_rows(ep, n_step=1, gamma=0.99, horizon=10)
        terminals = sum(1 for r in rows if r[4] == 0.0)
        real = int((ep['discount'][1:] == 0.0).sum())
        assert terminals == real, (
            f'{f.name}: {terminals} terminal rows for {real} real ones')


def test_the_achievement_reward_lands_on_the_state_that_earns_it():
    """The +1 must sit on the row whose own action unlocked something."""
    checked = 0
    for f in _demo_files():
        ep = np.load(f)
        rows = _episode_to_rows(ep, n_step=1, gamma=0.99, horizon=10)
        actions, rewards = ep['action'], ep['reward']
        for t in range(len(actions) - 1):
            if int(actions[t + 1]) in CRAFT_ACTIONS and rewards[t + 1] >= 0.9:
                row = rows[t]
                assert row[1] == int(actions[t + 1])
                assert row[2] >= 0.9, (
                    f'{f.name}: craft at {t + 1} earned {rewards[t + 1]} but '
                    f'its row carries G={row[2]}')
                checked += 1
    assert checked > 100, f'only {checked} rewarded crafts checked'
