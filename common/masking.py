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

# Material ids for the extended mask, same source as above.
WATER_ID = 1
GRASS_ID = 2
STONE_ID = 3
PATH_ID = 4
SAND_ID = 5
TREE_ID = 6
LAVA_ID = 7
COAL_ID = 8
IRON_ID = 9
DIAMOND_ID = 10

# Object ids. SemanticView numbers objects after the materials, in the order
# crafter/env.py passes them; verified by printing _obj_ids, not assumed.
PLAYER_ID = 13
COW_ID = 14
ZOMBIE_ID = 15
SKELETON_ID = 16
ARROW_ID = 17
PLANT_ID = 18

# data.yaml walkable:, plus lava -- Player.walkable adds it (objects.py:96),
# and walking into lava is legal, it just kills you. Masking it would be a
# different intervention (safety shaping), so it stays legal here.
WALKABLE_IDS = frozenset((GRASS_ID, PATH_ID, SAND_ID, LAVA_ID))

# What `do` can act on, from data.yaml collect: plus the objects
# _do_object handles. Value is the inventory requirement, from collect:
# require: -- empty dict means no tool needed.
_DO_MATERIAL_REQ = {
    TREE_ID: {},
    WATER_ID: {},
    GRASS_ID: {},
    STONE_ID: {'wood_pickaxe': 1},
    COAL_ID: {'wood_pickaxe': 1},
    IRON_ID: {'stone_pickaxe': 1},
    DIAMOND_ID: {'iron_pickaxe': 1},
}

# Objects `do` always does something to: damage, or eat a ripe plant. A plant
# that is not ripe yields nothing, but ripeness is not in the semantic map, so
# PLANT_ID stays legal -- permissive where it cannot be sure.
_DO_OBJECT_IDS = frozenset((COW_ID, ZOMBIE_ID, SKELETON_ID, PLANT_ID))

DO = 5
SLEEP = 6
ENERGY_MAX = 9      # data.yaml items: energy: max

# action -> facing offset, from Player._move's direction table
# (crafter/objects.py:174). Action order is crafter/constants.py actions:.
_MOVE_DIR = {
    1: (-1, 0),     # move_left
    2: (+1, 0),     # move_right
    3: (0, -1),     # move_up
    4: (0, +1),     # move_down
}

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


def _tile(semantic, pos, offset):
    """The semantic id of pos+offset, or None if it is off the map."""
    x = int(pos[0]) + offset[0]
    y = int(pos[1]) + offset[1]
    h, w = semantic.shape
    if not (0 <= x < h and 0 <= y < w):
        return None
    return int(semantic[x, y])


def _do_is_live(target, inv, skip_grass=False):
    """Would `do` on this tile id change anything?

    Permissive on the two things the semantic map cannot tell us: an unknown
    id, and a tile occupied by an object we do not model (the map paints
    objects over the material beneath, so an Arrow standing on a tree hides
    the tree).

    skip_grass also masks `do` at grass. Grass IS collectable -- data.yaml
    gives a sapling with probability 0.1 -- so strictly it is a legal press
    and the default leaves it alone. But the agent faces grass on ~84% of
    steps (measured over 10k random steps) while collect_sapling already sits
    at 91%, so those presses buy an achievement that is long since solved.
    This is a behavioural prior, not a legality rule; it is the one part of
    the extended mask that can forbid something that would have worked, so it
    is a separate flag and is reported separately.
    """
    if target is None:
        return True
    if target in _DO_OBJECT_IDS:
        return True
    # An unmodelled object: cannot see the material under it.
    if target >= FIRST_OBJECT_ID and target != PLAYER_ID:
        return True
    if skip_grass and target == GRASS_ID:
        return False
    req = _DO_MATERIAL_REQ.get(target)
    if req is None:
        # A material not in data.yaml collect: -- path, sand, table, furnace,
        # lava, or the 0 'unknown' id. `do` on it is a guaranteed no-op.
        return False
    # Collectable, but only with the right tool. This is the stone case: the
    # whole point of the rule is that `do` at stone with no wood pickaxe
    # returns early (objects.py:222) and teaches the agent nothing.
    return all(inv.get(k, 0) >= v for k, v in req.items())


def _extend_mask(mask, info, inv, sem, pos, skip_grass=False):
    """The three extra rules: `do`, `sleep`, movement. Mutates `mask`.

    Needs player_facing, which Crafter's own info does not carry -- common/env.py's
    VecEnv adds it. Without it the facing-dependent rules are skipped rather
    than guessed, so a caller on a bare crafter.Env keeps the base mask.
    """
    facing = info.get('player_facing')

    # -- rule 2: sleep. Sleeping at full energy does nothing at all
    # (objects.py:115 gates on energy < max), it just wastes the step.
    if inv.get('energy', 0) >= ENERGY_MAX:
        mask[SLEEP] = 0.0

    if sem is None or pos is None or facing is None:
        return mask
    sem = np.asarray(sem)

    # -- rule 1: `do` acts on the single tile the player faces.
    if not _do_is_live(_tile(sem, pos, facing), inv, skip_grass):
        mask[DO] = 0.0

    # -- rule 3: movement. THE TRAP. A move into a tile you cannot enter is
    # not a no-op: Player._move sets facing FIRST and then tries to step
    # (objects.py:174-177), so a blocked move TURNS you, which is how the
    # agent lines up a `do`. Masking every blocked move would make it unable
    # to aim at a tree it is standing beside.
    #
    # So mask a move only when it is BOTH blocked AND redundant as a turn --
    # i.e. the player already faces that way. Then it provably does nothing:
    # facing is already set and the step fails.
    for action, direction in _MOVE_DIR.items():
        if tuple(facing) != direction:
            continue                     # still useful as a turn
        target = _tile(sem, pos, direction)
        if target is None:
            continue                     # off-map: let the engine decide
        if target in WALKABLE_IDS:
            continue                     # the move would actually move us
        if target >= FIRST_OBJECT_ID and target != PLAYER_ID:
            # An object blocks the tile, but objects move. Keep it legal.
            continue
        mask[action] = 0.0

    return mask


def legal_mask(info, extended=False, skip_grass=False):
    """-> (17,) float32, 1.0 where the action could do something.

    With extended=False only the place/make actions are ever masked and
    actions 0-6 are always legal, which is the mask every run up to
    1m_fix_mask used. extended=True adds the `do`, `sleep` and movement rules
    -- see _extend_mask. It is opt-in so the old runs stay reproducible.

    Returns all-ones when info lacks the keys, so a caller holding a reset's
    empty info does not forbid everything.
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

    if extended:
        _extend_mask(mask, info, inv, sem, pos, skip_grass)

    return mask


def legal_masks(infos, dones=None, extended=False, skip_grass=False):
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
        masks[i] = legal_mask(info, extended=extended,
                              skip_grass=skip_grass)
    return masks


def random_legal(masks):
    """Uniform over each row's legal actions. -> (N,) int64."""
    out = np.zeros(len(masks), dtype=np.int64)
    for i, m in enumerate(masks):
        legal = np.flatnonzero(m > 0)
        out[i] = np.random.choice(legal) if len(legal) else 0
    return out
