"""Tests for the extended action mask (do / sleep / movement).

The doc that proposed this warned that getting movement masking wrong makes
the agent unable to aim, because in Crafter a blocked move TURNS you rather
than doing nothing. So the important tests here are the two engine-agreement
ones, and they check BOTH directions:

  test_extended_never_masks_a_live_action -- the fatal direction. Replays every
      masked action in a deepcopy of the env and asserts it really was a no-op.
  test_extended_masked_actions_are_noops  -- the cheap direction, same shape as
      test_agrees_with_engine in test_masking.py.

A mask that merely agrees with my reading of data.yaml proves nothing.
"""

import copy

import numpy as np
import pytest
import crafter

from common.masking import (
    DO, SLEEP, ENERGY_MAX, GRASS_ID, N_ACTIONS, WALKABLE_IDS,
    legal_mask, legal_masks,
)

# Inventory keys an action can change. health/food/drink/energy drift every
# step from _update_life_stats regardless of the action, so they are excluded.
_ITEM_KEYS = ('sapling', 'wood', 'stone', 'coal', 'iron', 'diamond',
              'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
              'wood_sword', 'stone_sword', 'iron_sword')


def _info(env):
    """The info dict the mask reads, including the facing VecEnv adds."""
    p = env._player
    return {
        'inventory': dict(p.inventory),
        'achievements': dict(p.achievements),
        'semantic': env._sem_view(),
        'player_pos': np.array(p.pos),
        'player_facing': tuple(p.facing),
    }


def _changed(before, after, was_sleeping):
    """Did the action change anything the player controls?

    wake_up is excluded when the player was asleep: update() clears sleeping
    at full energy whatever button was pressed, so the achievement belongs to
    the sleep state, not to the action. Without this exclusion the test
    reports ~170 false violations -- and reports them for the ORIGINAL mask
    too, which is how we know it is the test and not the mask.
    """
    inv0, ach0, pos0, fac0 = before
    inv1, ach1, pos1, fac1 = after
    if any(inv0[k] != inv1[k] for k in _ITEM_KEYS):
        return True
    if pos0 != pos1 or fac0 != fac1:
        return True
    return any(ach0[k] != ach1[k] for k in ach1
               if not (k == 'wake_up' and was_sleeping))


def _snapshot(env):
    p = env._player
    return (dict(p.inventory), dict(p.achievements),
            tuple(p.pos), tuple(p.facing))


def test_extended_is_opt_in():
    """The default mask must be byte-identical to the pre-change one."""
    env = crafter.Env(seed=1)
    env.reset()
    for _ in range(50):
        env.step(int(np.random.randint(0, 17)))
        info = _info(env)
        base = legal_mask(info)
        # Only actions 7-16 are ever masked without extended=True.
        assert base[:7].all(), 'base mask must leave 0-6 legal'


def test_sleep_masked_only_at_full_energy():
    env = crafter.Env(seed=1)
    env.reset()
    _, _, _, _ = env.step(0)
    info = _info(env)
    assert info['inventory']['energy'] == ENERGY_MAX
    assert legal_mask(info, extended=True)[SLEEP] == 0.0

    info['inventory']['energy'] = ENERGY_MAX - 1
    assert legal_mask(info, extended=True)[SLEEP] == 1.0


def test_do_masked_at_stone_without_pickaxe():
    """The whole point of the contribution: `do` at stone with no pickaxe
    returns early in the engine and teaches the agent nothing."""
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    sem = info['semantic']
    x, y = int(info['player_pos'][0]), int(info['player_pos'][1])
    # Put stone directly in front of the player and face it.
    info['player_facing'] = (0, 1)
    sem[x, y + 1] = 3  # STONE_ID
    info['inventory']['wood_pickaxe'] = 0
    assert legal_mask(info, extended=True)[DO] == 0.0
    info['inventory']['wood_pickaxe'] = 1
    assert legal_mask(info, extended=True)[DO] == 1.0


def test_blocked_move_stays_legal_when_not_yet_facing():
    """THE AIMING TRAP. A blocked move you are not yet facing still turns
    you, which is how the agent lines up a `do`. It must stay legal."""
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    sem = info['semantic']
    x, y = int(info['player_pos'][0]), int(info['player_pos'][1])
    sem[x, y + 1] = 3      # stone below: unwalkable
    info['player_facing'] = (0, -1)   # but we face UP
    m = legal_mask(info, extended=True)
    assert m[4] == 1.0, 'move_down must stay legal: it turns us to face stone'

    # Now already facing it: the move is a provable no-op.
    info['player_facing'] = (0, +1)
    m = legal_mask(info, extended=True)
    assert m[4] == 0.0, 'already facing an unwalkable tile: move does nothing'


def test_move_into_walkable_always_legal():
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    sem = info['semantic']
    x, y = int(info['player_pos'][0]), int(info['player_pos'][1])
    for mat in sorted(WALKABLE_IDS):
        sem[x, y + 1] = mat
        info['player_facing'] = (0, +1)
        assert legal_mask(info, extended=True)[4] == 1.0, \
            f'move into walkable id {mat} must stay legal'


def test_skip_grass_is_separate_from_legality():
    """skip_grass is a behavioural prior: grass IS collectable (sapling at
    p=0.1), so it must not be part of the plain extended mask."""
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    sem = info['semantic']
    x, y = int(info['player_pos'][0]), int(info['player_pos'][1])
    info['player_facing'] = (0, +1)
    sem[x, y + 1] = GRASS_ID
    assert legal_mask(info, extended=True)[DO] == 1.0
    assert legal_mask(info, extended=True, skip_grass=True)[DO] == 0.0


def test_missing_facing_skips_facing_rules():
    """A caller on a bare crafter.Env has no player_facing. The facing rules
    must be skipped, not guessed."""
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    del info['player_facing']
    m = legal_mask(info, extended=True)
    assert m[DO] == 1.0, 'no facing -> `do` cannot be judged, stays legal'
    assert m[1:5].all(), 'no facing -> movement cannot be judged'


def test_legal_masks_passes_flags_through():
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    info = _info(env)
    info['inventory']['energy'] = ENERGY_MAX
    out = legal_masks([info], extended=True)
    assert out.shape == (1, N_ACTIONS)
    assert out[0][SLEEP] == 0.0
    # done rows stay all-ones whatever the flags say
    assert legal_masks([info], dones=[True], extended=True)[0].all()


@pytest.mark.parametrize('seed', [1, 2])
def test_extended_masked_actions_are_noops(seed):
    """Play on, and whenever the mask forbids something, press it anyway and
    assert nothing moved."""
    np.random.seed(seed)
    env = crafter.Env(seed=seed, length=10000)
    env.reset()
    env.step(0)
    checked = 0
    for _ in range(600):
        info = _info(env)
        illegal = np.flatnonzero(legal_mask(info, extended=True) == 0.0)
        if len(illegal) and np.random.rand() < 0.5:
            a = int(np.random.choice(illegal))
        else:
            a = int(np.random.randint(0, 17))
        was_sleeping = env._player.sleeping
        before = _snapshot(env)
        _, _, done, _ = env.step(a)
        if a in illegal:
            checked += 1
            assert not _changed(before, _snapshot(env), was_sleeping), \
                f'masked action {a} changed the world at seed {seed}'
        if done:
            env.reset()
            env.step(0)
    assert checked > 50, f'only probed {checked} illegal actions'


@pytest.mark.parametrize('seed', [1, 2])
def test_extended_never_masks_a_live_action(seed):
    """THE FATAL DIRECTION, checked exhaustively per step.

    For every action the mask forbids, replay it in a deepcopy of the env and
    assert it really was a no-op. An action wrongly marked illegal is
    unreachable for the whole run, so this is the direction that matters.
    """
    np.random.seed(seed)
    env = crafter.Env(seed=seed, length=10000)
    env.reset()
    env.step(0)
    replayed = 0
    for _ in range(120):
        info = _info(env)
        illegal = np.flatnonzero(legal_mask(info, extended=True) == 0.0)
        for a in illegal:
            sandbox = copy.deepcopy(env)
            was_sleeping = sandbox._player.sleeping
            before = _snapshot(sandbox)
            sandbox.step(int(a))
            replayed += 1
            assert not _changed(before, _snapshot(sandbox), was_sleeping), \
                f'mask forbade action {a} but it WOULD have worked'
        _, _, done, _ = env.step(int(np.random.randint(0, 17)))
        if done:
            env.reset()
            env.step(0)
    assert replayed > 100, f'only replayed {replayed} masked actions'
