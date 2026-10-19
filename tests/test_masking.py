"""Tests for common/masking.py.

The important test is test_agrees_with_engine: it plays random episodes and
checks the mask against what the environment ACTUALLY did, by comparing the
inventory and achievement counters before and after each step. A mask that
merely agrees with my own reading of data.yaml proves nothing.
"""

import numpy as np
import pytest
import crafter

from common.masking import (
    N_ACTIONS, TABLE_ID, FURNACE_ID, ALWAYS_LEGAL,
    legal_mask, legal_masks, random_legal,
)


def _info_at_spawn(seed=1):
    env = crafter.Env(seed=seed)
    env.reset()
    _, _, _, info = env.step(0)
    return env, info


def test_shape_and_dtype():
    _, info = _info_at_spawn()
    m = legal_mask(info)
    assert m.shape == (N_ACTIONS,)
    assert m.dtype == np.float32
    assert set(np.unique(m)) <= {0.0, 1.0}


def test_movement_always_legal():
    _, info = _info_at_spawn()
    m = legal_mask(info)
    assert m[list(ALWAYS_LEGAL)].all(), 'noop/moves/do/sleep must never be masked'


def test_spawn_has_nothing_craftable():
    """Empty inventory at spawn, so every place and make is illegal."""
    _, info = _info_at_spawn()
    assert sum(info['inventory'][k] for k in
               ('wood', 'stone', 'coal', 'iron', 'sapling')) == 0
    m = legal_mask(info)
    assert not m[7:].any(), f'nothing should be craftable at spawn, got {m[7:]}'


def test_missing_info_is_permissive():
    """A reset gives no info. Forbidding everything there would be fatal."""
    for bad in (None, {}, {'achievements': {}}):
        assert legal_mask(bad).all()


def test_place_costs():
    _, info = _info_at_spawn()
    inv = info['inventory']

    inv['wood'] = 1
    assert legal_mask(info)[8] == 0.0, 'place_table needs 2 wood, not 1'
    inv['wood'] = 2
    assert legal_mask(info)[8] == 1.0

    inv['stone'] = 3
    m = legal_mask(info)
    assert m[7] == 1.0, 'place_stone needs 1 stone'
    assert m[9] == 0.0, 'place_furnace needs 4 stone, not 3'
    inv['stone'] = 4
    assert legal_mask(info)[9] == 1.0

    inv['sapling'] = 1
    assert legal_mask(info)[10] == 1.0


def test_make_needs_table():
    """1 wood is not enough for a wood pickaxe -- a table must be adjacent."""
    env, info = _info_at_spawn()
    info['inventory']['wood'] = 5
    assert legal_mask(info)[11] == 0.0, 'no table nearby yet'

    x, y = info['player_pos']
    info['semantic'][x, y + 1] = TABLE_ID
    m = legal_mask(info)
    assert m[11] == 1.0, 'wood pickaxe: 1 wood + table'
    assert m[14] == 1.0, 'wood sword: 1 wood + table'
    assert m[12] == 0.0, 'stone pickaxe also needs stone'
    assert m[16] == 0.0, 'iron sword also needs coal, iron and a furnace'


def test_make_iron_needs_furnace_too():
    env, info = _info_at_spawn()
    info['inventory'].update(wood=1, coal=1, iron=1)
    x, y = info['player_pos']
    info['semantic'][x, y + 1] = TABLE_ID
    assert legal_mask(info)[13] == 0.0, 'iron pickaxe needs a furnace as well'
    info['semantic'][x, y - 1] = FURNACE_ID
    m = legal_mask(info)
    assert m[13] == 1.0 and m[16] == 1.0


def test_table_beyond_one_tile_does_not_count():
    """nearby() is distance 1. Two tiles away must not enable crafting."""
    env, info = _info_at_spawn()
    info['inventory']['wood'] = 5
    x, y = info['player_pos']
    info['semantic'][x, y + 2] = TABLE_ID
    assert legal_mask(info)[11] == 0.0


def test_patch_clipped_at_map_edge():
    """A player on the boundary must not wrap or raise."""
    _, info = _info_at_spawn()
    info['inventory']['wood'] = 5
    for pos in ([0, 0], [63, 63], [0, 63]):
        info['player_pos'] = np.array(pos)
        m = legal_mask(info)
        assert m.shape == (N_ACTIONS,)


def test_object_occlusion_is_permissive():
    """SemanticView paints objects over materials, hiding a table beneath.

    The engine's nearby() reads _mat_map and still sees it, so the mask must
    not forbid crafting just because a creature in the 3x3 might be standing
    on a table.
    """
    _, info = _info_at_spawn()
    info['inventory']['wood'] = 5
    x, y = info['player_pos']
    sem = info['semantic']
    sem[x-1:x+2, y-1:y+2] = 2        # all grass, no table
    assert legal_mask(info)[11] == 0.0
    sem[x, y + 1] = 14               # a cow beside us, maybe on a table
    assert legal_mask(info)[11] == 1.0, 'an object may be hiding a table'


def test_own_tile_does_not_trigger_occlusion():
    """The player is always on its own tile, so it must not mean 'maybe'.

    If it did, the occlusion rule would fire every step and the table
    requirement would never mask anything at all.
    """
    _, info = _info_at_spawn()
    info['inventory']['wood'] = 5
    x, y = info['player_pos']
    sem = info['semantic']
    sem[x-1:x+2, y-1:y+2] = 2
    sem[x, y] = 13                   # the player, as the real env draws it
    assert legal_mask(info)[11] == 0.0, 'own tile is known, not unknown'


def test_legal_masks_ignores_stale_info_on_done():
    """VecEnv returns the dying step's info with the next episode's obs."""
    _, info = _info_at_spawn()
    infos = [info, info]
    masked = legal_masks(infos, dones=[False, True])
    assert not masked[0][7:].any(), 'live env: spawn has nothing craftable'
    assert masked[1].all(), 'done env: stale info must not constrain the reset'
    # Without dones it must not silently do the right thing by accident.
    assert not legal_masks(infos)[1][7:].any()


def test_random_legal_only_picks_legal():
    masks = np.zeros((4, N_ACTIONS), dtype=np.float32)
    masks[0, 5] = 1.0
    masks[1, [0, 16]] = 1.0
    masks[2] = 1.0
    # row 3 left all-zero: degenerate, must not crash
    for _ in range(200):
        a = random_legal(masks)
        assert a[0] == 5
        assert a[1] in (0, 16)
        assert 0 <= a[2] < N_ACTIONS
        assert a[3] == 0
    assert a.dtype == np.int64


def test_random_legal_covers_all_legal_actions():
    masks = np.ones((1, N_ACTIONS), dtype=np.float32)
    seen = {int(random_legal(masks)[0]) for _ in range(2000)}
    assert seen == set(range(N_ACTIONS))


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_agrees_with_engine(seed):
    """Ground truth: a masked-out action must provably do nothing.

    Plays random actions, and for every step the mask called illegal, asserts
    the env's inventory and achievement counters did not move. This is the
    test that would catch a wrong material id or an off-by-one in the patch.
    """
    env = crafter.Env(seed=seed, length=400)
    env.reset()
    _, _, _, info = env.step(0)

    checked = 0
    for _ in range(400):
        mask = legal_mask(info)
        illegal = np.flatnonzero(mask == 0.0)
        if len(illegal):
            action = int(np.random.choice(illegal))
            inv_before = dict(env._player.inventory)
            ach_before = dict(env._player.achievements)
            _, _, done, info_after = env.step(action)
            # Only crafting actions are ever masked, and those touch the
            # inventory items -- not health/food/drink/energy, which drift
            # every step on their own.
            items = ('wood', 'stone', 'coal', 'iron', 'sapling', 'diamond',
                     'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                     'wood_sword', 'stone_sword', 'iron_sword')
            for k in items:
                assert inv_before[k] == env._player.inventory[k], (
                    f'action {action} was masked illegal but changed {k}')
            assert ach_before == dict(env._player.achievements), (
                f'action {action} was masked illegal but unlocked something')
            checked += 1
        else:
            _, _, done, info_after = env.step(np.random.randint(N_ACTIONS))
        info = info_after
        if done:
            env.reset()
            _, _, _, info = env.step(0)
    assert checked > 50, f'only exercised {checked} illegal actions'
