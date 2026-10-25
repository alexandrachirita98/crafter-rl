"""The aux label must see the unlock the state's own action causes.

The loop passed info['achievements'], which is the count AFTER the action.
_label compares the future against that base, so an unlock caused by the
action itself was already in the base and invisible: the state whose action
collects the wood was labelled "none soon", and the state before it got the
label. The head still trained, accuracy still beat the majority class, and
the signal was blurred exactly where it should be sharpest.

See docs/performance-issue-survey.md section 4.
"""

import numpy as np

from common.metrics import ACHIEVEMENTS
from stage_11_c1_achievement_aux.train import (AchievementLabeller, N_ACH,
                                               NONE_SOON)

WOOD = ACHIEVEMENTS.index('collect_wood')
TABLE = ACHIEVEMENTS.index('place_table')


def _run_loop(unlock_at, horizon=10, steps=40, unlock_index=WOOD):
    """Replay the loop's label bookkeeping: counts are held back one step.

    Returns {state_name: label}. `unlock_at` is the step whose ACTION earns
    the achievement, so info after that step shows the rise.
    """
    lab = AchievementLabeller(horizon=horizon)
    labels = {}
    counts = np.zeros(N_ACH, dtype=np.int32)      # the counts AT state s0
    info_counts = np.zeros(N_ACH, dtype=np.int32)
    for t in range(steps):
        for payload, label in lab.push(f's{t}', counts):
            labels[payload] = label
        if t == unlock_at:
            info_counts = info_counts.copy()
            info_counts[unlock_index] += 1
        counts = info_counts.copy()               # base for s{t+1}
    for payload, label in lab.flush():
        labels[payload] = label
    return labels


def test_the_unlocking_state_is_labelled_with_its_own_unlock():
    """THE regression. s3's action collects the wood, so s3 sees it coming."""
    labels = _run_loop(unlock_at=3)
    assert labels['s3'] == WOOD, (
        f's3 labelled {labels["s3"]}, want collect_wood ({WOOD}) -- the state '
        'whose own action unlocks must not read as "none soon"')


def test_states_before_the_unlock_also_see_it():
    labels = _run_loop(unlock_at=3)
    assert labels['s0'] == WOOD
    assert labels['s2'] == WOOD


def test_states_after_the_unlock_do_not_see_it_again():
    """Counts are cumulative, so a past unlock must not keep labelling."""
    labels = _run_loop(unlock_at=3)
    assert labels['s4'] == NONE_SOON
    assert labels['s10'] == NONE_SOON


def test_the_horizon_is_respected():
    """The lookahead is bounded, and the boundary is sharp.

    With the base held back one step, the deepest state that still sees an
    unlock at step k is s_k itself (its own action causes it) and the
    shallowest is s_{k-horizon+1}. Measured, not assumed: at horizon 10 with
    the unlock at step 12 the labelled states are s3..s12 inclusive.
    """
    labels = _run_loop(unlock_at=12, horizon=10)
    assert labels['s3'] == WOOD            # exactly at the horizon edge
    assert labels['s2'] == NONE_SOON       # one step too far back
    assert labels['s12'] == WOOD           # the unlocking state itself


def test_a_second_achievement_is_labelled_independently():
    labels = _run_loop(unlock_at=5, unlock_index=TABLE)
    assert labels['s5'] == TABLE
    assert labels['s6'] == NONE_SOON


def test_counts_reset_at_an_episode_boundary():
    """A new world starts from zero, or the dead player's totals mask it.

    The loop zeroes `counts` on done for this reason; without it the next
    episode's base is the previous player's cumulative counts, so every
    achievement it already had reads as "already unlocked" and can never be
    labelled again.
    """
    lab = AchievementLabeller(horizon=5)
    # episode 1 ends having collected wood
    carried = np.zeros(N_ACH, dtype=np.int32)
    carried[WOOD] = 1
    lab.flush()                       # end_episode flushes the labeller

    # episode 2: fresh player, wood collected again by s1's action
    labels = {}
    counts = np.zeros(N_ACH, dtype=np.int32)      # NOT `carried`
    info_counts = np.zeros(N_ACH, dtype=np.int32)
    for t in range(10):
        for payload, label in lab.push(f'e2s{t}', counts):
            labels[payload] = label
        if t == 1:
            info_counts = info_counts.copy()
            info_counts[WOOD] += 1
        counts = info_counts.copy()
    for payload, label in lab.flush():
        labels[payload] = label
    assert labels['e2s1'] == WOOD, (
        'the new episode cannot see its own first unlock; counts carried '
        'across the seam')
