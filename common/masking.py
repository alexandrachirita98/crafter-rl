"""Legal-action masking for Crafter.

Reads the inventory and semantic map out of info to decide which of the 17
actions can do anything right now. This is privileged state -- none of it is
visible in the 64x64 image the agent sees -- so a masked run is not strictly
comparable to the pixels-only numbers in the Crafter paper. Say so in the
write-up.

Rules come from crafter/data.yaml (place:/make:). 'nearby' is distance 1, the
3x3 square around the player -- crafter/objects.py:252 calls
world.nearby(self.pos, 1).

The mask is permissive where it cannot be sure. An action marked legal that
turns out to be a no-op costs one wasted step; an action wrongly marked
illegal is unreachable for the whole run. The two errors are not symmetric,
so every unknown resolves to legal.
"""

import numpy as np

N_ACTIONS = 17

# Semantic-map ids, from World._mat_ids via SemanticView (crafter/engine.py:255).
# Verified by running the env, not read off the materials list -- the map is
# 1-indexed because _mat_ids starts at {None: 0}.
TABLE_ID = 11
FURNACE_ID = 12

# Ids >= this are objects, not materials. SemanticView paints objects over the
# material tile they stand on (engine.py:263), so these hide what is beneath.
FIRST_OBJECT_ID = 13

# action -> inventory cost, from data.yaml place:
_PLACE_COST = {
    7: {'stone': 1},
    8: {'wood': 2},
    9: {'stone': 4},
    10: {'sapling': 1},
}

# action -> (inventory cost, needs_table, needs_furnace), from data.yaml make:
_MAKE_REQ = {
    11: ({'wood': 1}, True, False),
    12: ({'wood': 1, 'stone': 1}, True, False),
    13: ({'wood': 1, 'coal': 1, 'iron': 1}, True, True),
    14: ({'wood': 1}, True, False),
    15: ({'wood': 1, 'stone': 1}, True, False),
    16: ({'wood': 1, 'coal': 1, 'iron': 1}, True, True),
}

ALWAYS_LEGAL = (0, 1, 2, 3, 4, 5, 6)  # noop, four moves, do, sleep


def _patch(semantic, pos):
    """The 3x3 of the semantic map around pos, player's own tile blanked.

    Clipped at the map edges. The centre tile is set to 0 because the player
    is always drawn there (SemanticView paints objects over the material they
    stand on, engine.py:263) and so tells us nothing -- blanking it keeps the
    occlusion rule below from firing on every single step.
    """
    x, y = int(pos[0]), int(pos[1])
    h, w = semantic.shape
    x0, y0 = max(0, x - 1), max(0, y - 1)
    patch = semantic[x0:min(h, x + 2), y0:min(w, y + 2)].copy()
    patch[x - x0, y - y0] = 0
    return patch


def _nearby(patch, mat_id):
    """Is mat_id in the patch, or might another object be hiding one?

    A cow, zombie or plant standing on a table hides it from the semantic map
    while the engine's own nearby() -- which reads _mat_map -- still sees it.
    We cannot see underneath, so a non-player object in the 3x3 counts as
    'maybe' and we answer yes. That costs an occasional wasted craft press;
    the opposite error would make crafting unreachable. The player's own tile
    is already blanked by _patch, so it does not trigger this.
    """
    if (patch == mat_id).any():
        return True
    return bool((patch >= FIRST_OBJECT_ID).any())


def legal_mask(info):
    """-> (17,) float32, 1.0 where the action could do something.

    Actions 0-6 are always legal. Returns all-ones when info lacks the keys,
    so a caller holding a reset's empty info does not forbid everything.
    """
    mask = np.ones(N_ACTIONS, dtype=np.float32)
    if not info or 'inventory' not in info:
        return mask

    inv = info['inventory']
    sem = info.get('semantic')
    pos = info.get('player_pos')

    for action, cost in _PLACE_COST.items():
        if any(inv.get(k, 0) < v for k, v in cost.items()):
            mask[action] = 0.0

    # No map means no idea what is nearby -- assume the utility is there.
    has_table = has_furnace = True
    if sem is not None and pos is not None:
        patch = _patch(np.asarray(sem), pos)
        has_table = _nearby(patch, TABLE_ID)
        has_furnace = _nearby(patch, FURNACE_ID)

    for action, (cost, need_table, need_furnace) in _MAKE_REQ.items():
        if any(inv.get(k, 0) < v for k, v in cost.items()):
            mask[action] = 0.0
        elif need_table and not has_table:
            mask[action] = 0.0
        elif need_furnace and not has_furnace:
            mask[action] = 0.0

    return mask


def legal_masks(infos, dones=None):
    """Stack masks for a VecEnv step. -> (N, 17) float32.

    dones matters. VecEnv.step auto-resets and returns the DYING step's info
    alongside the NEXT episode's first observation (common/env.py:113), so the
    info describes a world that no longer exists -- the dead player's 5 wood,
    not the fresh player's nothing. Masking the new state with it would forbid
    legal actions for a whole episode's first step. Where done is True we fall
    back to all-ones, which is also what reset() gives us (reset returns no
    info at all).
    """
    masks = np.ones((len(infos), N_ACTIONS), dtype=np.float32)
    for i, info in enumerate(infos):
        if dones is not None and dones[i]:
            continue
        masks[i] = legal_mask(info)
    return masks


def random_legal(masks):
    """Uniform over each row's legal actions. -> (N,) int64."""
    out = np.zeros(len(masks), dtype=np.int64)
    for i, m in enumerate(masks):
        legal = np.flatnonzero(m > 0)
        out[i] = np.random.choice(legal) if len(legal) else 0
    return out
