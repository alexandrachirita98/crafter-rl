"""Scripted expert that records Crafter demonstrations headlessly.

Replaces a human sitting at crafter/run_gui.py. Writes the SAME .npz files
via crafter.Recorder, so the stage 12 demo loader reads these with no changes.

Needs no display and no pygame. Run it anywhere:

    python record_demos.py --episodes 40 --outdir demos

The expert is a hand-written finite-state planner, not a learned agent. It
reads info['semantic'] (the full 64x64 world map) and info['inventory'],
picks the next tech-tree goal it can make progress on, and BFS-paths to the
nearest tile that serves that goal. This is deliberately "cheating" in a way
a learning agent never could -- which is the point: a demonstration only has
to be good, not learnable.
"""
import argparse
import collections
import datetime
import json
import pathlib

import numpy as np

import crafter

# --- semantic map ids, from crafter/engine.py SemanticView -------------------
MAT = {'none': 0, 'water': 1, 'grass': 2, 'stone': 3, 'path': 4, 'sand': 5,
       'tree': 6, 'lava': 7, 'coal': 8, 'iron': 9, 'diamond': 10,
       'table': 11, 'furnace': 12}
OBJ = {'player': 13, 'cow': 14, 'zombie': 15, 'skeleton': 16, 'arrow': 17,
       'plant': 18}
WALKABLE = {MAT['grass'], MAT['path'], MAT['sand']}
# Stone held back purely for building a shelter to sleep in. Sleeping is the
# only way to restore energy, and sleeping unsheltered costs 7 health a hit.
SHELTER_STONE = 4
# Start building the night shelter at energy 7, not at 5. Energy drains ~1 per
# 25 steps awake (objects.py:142), so 7 leaves ~50 steps to mine four stone and
# wall in -- enough, and still late enough not to interrupt the tech tree.
NIGHT_PREP_ENERGY = 7
NIGHT_ENERGY = 5
# VERIFIED, do not change: crafter/objects.py:252 calls world.nearby(pos, 1),
# so a table counts as "nearby" only within a 3x3 block centred on the player.
# Guessing a larger radius here makes the expert try to craft while out of
# range and it never completes a wood pickaxe.
TABLE_RADIUS = 1
# Lava is INSTANT DEATH (crafter/objects.py:178 sets health = 0), yet the engine
# lists it as walkable (objects.py:97) so the player can step straight in. The
# single biggest cause of short demo episodes. Never path through it, and never
# stand adjacent to it if there is any alternative.
LETHAL = {MAT['lava']}

# --- action indices, from crafter/constants.py actions ----------------------
A = {name: i for i, name in enumerate(crafter.constants.actions)}
# facing delta -> move action.  Crafter's axes: pos is (x, y), semantic is
# indexed [x, y], and 'move_up' decreases y.
DIR_TO_ACTION = {(-1, 0): A['move_left'], (1, 0): A['move_right'],
                 (0, -1): A['move_up'], (0, 1): A['move_down']}


def bfs_step(sem, start, targets, max_nodes=8000):
    """One move action toward the nearest tile in `targets`.

    Walks only over walkable tiles, but accepts standing ADJACENT to a target
    (you mine a tree by facing it, not by standing on it). Returns
    (action, reached_and_facing_delta) or (None, None) if unreachable.
    """
    if not targets:
        return None, None
    target_set = set(map(tuple, targets))
    start = tuple(start)
    seen = {start}
    # queue holds (pos, first_move_delta)
    queue = collections.deque([(start, None)])
    nodes = 0
    while queue and nodes < max_nodes:
        pos, first = queue.popleft()
        nodes += 1
        for delta, _ in DIR_TO_ACTION.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < sem.shape[0] and 0 <= nxt[1] < sem.shape[1]):
                continue
            step_first = first if first is not None else delta
            if nxt in target_set:
                # Adjacent to the goal. If we are already standing next to it,
                # just face it; otherwise take the first step of the path.
                if pos == start:
                    return None, delta
                return DIR_TO_ACTION[step_first], None
            if nxt in seen or sem[nxt] not in WALKABLE:
                continue
            seen.add(nxt)
            # Refuse to route through a tile that touches lava: a stray
            # re-plan one step later can walk us in.
            if _touches_lava(sem, nxt):
                continue
            queue.append((nxt, step_first))
    return None, None


def _touches_lava(sem, pos):
    x, y = pos
    lo_x, hi_x = max(0, x - 1), min(sem.shape[0], x + 2)
    lo_y, hi_y = max(0, y - 1), min(sem.shape[1], y + 2)
    return bool((sem[lo_x:hi_x, lo_y:hi_y] == MAT['lava']).any())


def find(sem, ids):
    """All coordinates whose semantic id is in `ids`."""
    mask = np.isin(sem, list(ids))
    return list(zip(*np.nonzero(mask)))


def near(sem, pos, ids, radius=1):
    """Is any tile with an id in `ids` within Chebyshev `radius`?"""
    x, y = pos
    lo_x, hi_x = max(0, x - radius), min(sem.shape[0], x + radius + 1)
    lo_y, hi_y = max(0, y - radius), min(sem.shape[1], y + radius + 1)
    return bool(np.isin(sem[lo_x:hi_x, lo_y:hi_y], list(ids)).any())


class ScriptedExpert:
    """Finite-state tech-tree planner.

    Goal order mirrors the Crafter tech tree. Each goal is a (name, condition,
    plan) triple; the first goal whose condition is unmet gets executed.
    """

    def __init__(self, rng):
        self.rng = rng
        self.facing = (0, 1)
        self.stuck = 0
        self.last_pos = None
        self.recent = collections.deque(maxlen=8)   # for oscillation detection

    def act(self, info):
        sem = info['semantic']
        pos = tuple(int(v) for v in info['player_pos'])
        inv = info['inventory']
        ach = info['achievements']

        # Anti-stall: if we have not moved in a while, take a random walk so a
        # bad BFS target cannot freeze the episode.
        if pos == self.last_pos:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pos = pos
        self.recent.append(pos)
        # Oscillation guard. The failure mode is not standing still, it is
        # pacing between two tiles: a re-plan each step flips the chosen
        # direction and the expert shuttles left-right forever with the
        # materials already in hand. Detect "visited <=2 distinct tiles over
        # the last 8 steps" and break out with a committed random walk.
        oscillating = (len(self.recent) == self.recent.maxlen
                       and len(set(self.recent)) <= 2)
        if self.stuck > 12 or oscillating:
            self.stuck = 0
            self.recent.clear()
            self.commit = self._safe_random_move(sem, pos)
            self.commit_left = 6
        # The commit may not outrank survival. Measured on seeds 3000-3002, a
        # random walk held the controls for 10-37 low-energy steps and 6-10
        # steps with an enemy adjacent -- the expert walked in circles while
        # being eaten. Drop the commit the moment either is true.
        # Only an actual threat cancels the commit. Low energy alone must not:
        # energy stays low for the whole night, so cancelling on it left the
        # oscillation guard permanently disabled and the expert paced between
        # two tiles until dawn (seed 3001, steps 170-203).
        urgent = (inv['health'] <= 5
                  or near(sem, pos, {OBJ['zombie'], OBJ['skeleton']}, 2))
        if getattr(self, 'commit_left', 0) > 0:
            if urgent:
                self.commit_left = 0
            else:
                self.commit_left -= 1
                return self.commit

        # --- survival first: these interrupt the tech tree ------------------
        # Health does not regenerate while any need is starved, and an episode
        # that dies at step 250 cannot reach the iron tools. Survival therefore
        # outranks every tech-tree goal, and the thresholds are generous.
        # Zombie mechanics, from crafter/objects.py:294-312: a zombie chases
        # from distance 8, hits for 2 while you are awake and **7 while you
        # sleep**, on a 5-step cooldown. Skeletons shoot arrows. So "is
        # anything within 1 tile" is far too late a question.
        adjacent = near(sem, pos, {OBJ['zombie'], OBJ['skeleton']}, 1)
        hunting = near(sem, pos, {OBJ['zombie'], OBJ['skeleton']}, 8)

        # Combat policy. Measured over 16 seeds, 13 of 16 deaths were enemies
        # landing repeated 2-damage hits while the expert stood and traded --
        # a zombie has 5 health and hits every 5 steps (objects.py:287-312),
        # so a swordless trade is a losing race. Decide by health, not by
        # proximity alone.
        armed = inv['stone_sword'] >= 1 or inv['wood_sword'] >= 1
        healthy = inv['health'] >= 4

        if adjacent and armed and healthy:
            # Only trade blows when we can win it. defeat_zombie and
            # defeat_skeleton are achievements worth having.
            for enemy in (OBJ['zombie'], OBJ['skeleton']):
                act = self._goto_and_do(sem, pos, {enemy}, only_if_adjacent=True)
                if act is not None:
                    return act

        # Hurt or unarmed with something hunting us: break contact. A stone
        # wall is the best answer because it blocks pathing outright.
        if adjacent and (not healthy or not armed):
            if inv['stone'] >= 1:
                act = self._seal_in(sem, pos)
                if act is not None:
                    return act
            return self._flee(sem, pos, {OBJ['zombie'], OBJ['skeleton']})

        # Adjacent but unarmed: never stand and take hits.
        if adjacent and not armed:
            return self._flee(sem, pos, {OBJ['zombie'], OBJ['skeleton']})

        # Thresholds are deliberately high. Health only regenerates when food,
        # drink and energy are ALL above zero (objects.py:153-164), so letting
        # any of them run low both stops healing and starts draining. Topping
        # up at 6 costs a short detour; running dry costs the episode.
        if inv['drink'] <= 6:
            act = self._goto_and_do(sem, pos, {MAT['water']})
            if act is not None:
                return act
        if inv['food'] <= 6:
            act = self._goto_and_do(sem, pos, {OBJ['cow']})
            if act is not None:
                return act
        # Sleep restores energy AND lets health regenerate, and wake_up is an
        # achievement. But sleeping next to a zombie takes 7 damage per hit
        # instead of 2, which is most of a full health bar. So: only sleep with
        # nothing hunting us within its own chase radius of 8, and wall in
        # first when something is out there.
        # Sleeping is NOT a single action: it persists for dozens of steps
        # while energy refills at +1 per ~10 ticks (objects.py:142-148), and a
        # zombie that was 8 tiles away when we lay down can reach us mid-sleep.
        # So an empty radius is not good enough -- we only sleep when sealed in,
        # meaning every walkable neighbour has been blocked with stone.
        # Stock the shelter reserve while it is still cheap. Waiting until
        # energy is already low means mining stone in the dark with zombies
        # spawning on every grass tile (env.py:144, target rises as daylight
        # falls). Four stone bought early is a survivable night.
        # Only stock while we are not already walling in. Mining is what
        # un-places a wall: without this guard the expert placed a stone,
        # saw stone < SHELTER_STONE, and mined the same wall straight back
        # out -- a place/mine loop, measured at seed 3001 steps 164-169.
        if (inv['stone'] < SHELTER_STONE and inv['energy'] <= NIGHT_PREP_ENERGY
                and inv['wood_pickaxe'] >= 1
                and self._open_neighbours(sem, pos) >= 3):
            act = self._goto_and_do(sem, pos, {MAT['stone']})
            if act is not None:
                return act

        if inv['energy'] <= NIGHT_ENERGY:
            if self._is_sheltered(sem, pos):
                return A['sleep']
            # Seal ourselves in with stone, one wall per step. Four stone buys
            # a full safe night, which is why SHELTER_STONE is reserved above.
            if inv['stone'] >= 1:
                act = self._seal_in(sem, pos)
                if act is not None:
                    return act
            if not hunting and self._open_neighbours(sem, pos) <= 1:
                return A['sleep']        # a natural dead-end is shelter enough
            # Out of stone. Mine some BEFORE fleeing, otherwise we ping-pong
            # between fleeing and seeking and starve on the spot.
            act = self._goto_and_do(sem, pos, {MAT['stone']})
            if act is not None:
                return act
            if hunting:
                return self._flee(sem, pos, {OBJ['zombie'], OBJ['skeleton']})
            if inv['energy'] <= 1:
                return A['sleep']        # out of options; sleeping beats starving

        # --- tech tree, in dependency order ---------------------------------
        # 1. wood
        if inv['wood'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['tree']})
            if act is not None:
                return act
        # 2. table (costs 2 wood)
        if not ach['place_table']:
            if inv['wood'] >= 2:
                return A['place_table']
            act = self._goto_and_do(sem, pos, {MAT['tree']})
            if act is not None:
                return act
        # 3. wood pickaxe and sword (1 wood each, needs table nearby)
        if not ach['make_wood_pickaxe']:
            return self._make(sem, pos, inv, 'wood_pickaxe', A['make_wood_pickaxe'])
        if not ach['make_wood_sword']:
            return self._make(sem, pos, inv, 'wood_sword', A['make_wood_sword'])
        # 4. sapling and plant (cheap, two achievements)
        if not ach['collect_sapling']:
            act = self._goto_and_do(sem, pos, {MAT['grass']})
            if act is not None:
                return act
        if not ach['place_plant'] and inv['sapling'] >= 1:
            act = self._goto_and_place(sem, pos, {MAT['grass']}, A['place_plant'])
            if act is not None:
                return act
        # 5. stone. Just get ONE to unlock collect_stone, then move straight
        # on to the tools -- each of which fetches its own materials via
        # _make. An earlier version demanded a large stockpile here
        # (goal needs + a shelter reserve = 7) before letting the chain
        # proceed, while the place_stone branch below spent a stone every time
        # it had one. The two fought: the stock never reached 7, the chain
        # never advanced, and make_stone_pickaxe sat at 0 for every seed.
        # Lesson that generalises: a planner must never have one rule saving a
        # resource and another spending it, with no arbiter between them.
        if inv['stone'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['stone']})
            if act is not None:
                return act

        # 6. stone tools FIRST (1 wood + 1 stone each). These are achievements;
        # place_stone is also an achievement but costs stone the tools need, so
        # it waits until the tools are done.
        if not ach['make_stone_pickaxe']:
            return self._make(sem, pos, inv, 'stone_pickaxe', A['make_stone_pickaxe'],
                              needs={'wood': 1, 'stone': 1})
        if not ach['make_stone_sword']:
            return self._make(sem, pos, inv, 'stone_sword', A['make_stone_sword'],
                              needs={'wood': 1, 'stone': 1})

        # 7. now spend a stone on place_stone, and stock up for the furnace.
        # Both spends go through _spare_stone: the shelter reserve is the one
        # resource with no substitute, because without it the expert cannot
        # sleep and the first night ends the episode. Spending down to zero
        # here was the bug that capped every demo at stone tier.
        if not ach['place_stone'] and self._spare_stone(inv) >= 1:
            act = self._goto_and_place(sem, pos, WALKABLE, A['place_stone'])
            if act is not None:
                return act
        if not ach['place_furnace']:
            if self._spare_stone(inv) >= 1:
                act = self._goto_and_place(sem, pos, WALKABLE, A['place_furnace'])
                if act is not None:
                    return act
            act = self._goto_and_do(sem, pos, {MAT['stone']})
            if act is not None:
                return act
        # 8. coal and iron (need the wood/stone pickaxe we now have)
        if inv['coal'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['coal']})
            if act is not None:
                return act
        if inv['iron'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['iron']})
            if act is not None:
                return act
        # 9. iron tools: need wood + coal + iron, AND table + furnace nearby
        if not ach['make_iron_pickaxe']:
            return self._make_iron(sem, pos, inv, A['make_iron_pickaxe'])
        if not ach['make_iron_sword']:
            return self._make_iron(sem, pos, inv, A['make_iron_sword'])
        # 10. diamond (needs iron pickaxe)
        if not ach['collect_diamond']:
            act = self._goto_and_do(sem, pos, {MAT['diamond']})
            if act is not None:
                return act
        # 11. eat_plant: a plant we placed earlier may now be ripe
        if not ach['eat_plant']:
            act = self._goto_and_do(sem, pos, {OBJ['plant']})
            if act is not None:
                return act

        # Everything done or unreachable: wander so the episode keeps logging.
        if self.rng.random() < 0.3:
            return A['do']
        return self._safe_random_move(sem, pos)

    # --- primitives ---------------------------------------------------------
    def _safe_random_move(self, sem, pos):
        """A random move that never steps into lava or off the map."""
        options = []
        for delta, action in DIR_TO_ACTION.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < sem.shape[0] and 0 <= nxt[1] < sem.shape[1]):
                continue
            if sem[nxt] in LETHAL:
                continue
            options.append(action)
        if not options:
            return A['noop']
        return int(self.rng.choice(options))

    def _spare_stone(self, inv):
        """Stone available for achievements, after the shelter reserve.

        The reserve is only held once we can actually use it -- before the
        wood pickaxe there is no mining, and holding back stone we do not have
        would just stall the chain.
        """
        if inv['wood_pickaxe'] < 1:
            return inv['stone']
        return max(0, inv['stone'] - SHELTER_STONE)

    def _open_neighbours(self, sem, pos):
        n = 0
        for delta in DIR_TO_ACTION:
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < sem.shape[0] and 0 <= nxt[1] < sem.shape[1]):
                continue
            if sem[nxt] in WALKABLE:
                n += 1
        return n

    def _is_sheltered(self, sem, pos):
        """Sealed in: nothing can walk to us, so sleeping is safe."""
        return self._open_neighbours(sem, pos) == 0

    def _seal_in(self, sem, pos):
        """Place stone into an open neighbour, facing it first if needed."""
        for delta, move in DIR_TO_ACTION.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < sem.shape[0] and 0 <= nxt[1] < sem.shape[1]):
                continue
            if sem[nxt] not in WALKABLE:
                continue
            if self.facing != delta:
                self.facing = delta
                return move
            return A['place_stone']
        return None

    def _flee(self, sem, pos, enemy_ids):
        """Move to whichever neighbour maximises distance from the nearest enemy."""
        enemies = find(sem, enemy_ids)
        if not enemies:
            return self._safe_random_move(sem, pos)
        best, best_action = -1.0, A['noop']
        for delta, action in DIR_TO_ACTION.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if not (0 <= nxt[0] < sem.shape[0] and 0 <= nxt[1] < sem.shape[1]):
                continue
            if sem[nxt] not in WALKABLE or sem[nxt] in LETHAL:
                continue
            d = min(abs(nxt[0] - e[0]) + abs(nxt[1] - e[1]) for e in enemies)
            if d > best:
                best, best_action = d, action
        return best_action

    def _goto_and_do(self, sem, pos, ids, only_if_adjacent=False):
        targets = find(sem, ids)
        if not targets:
            return None
        if only_if_adjacent and not near(sem, pos, ids, 1):
            return None
        move, face = bfs_step(sem, pos, targets)
        if face is not None:
            # Adjacent already. Face it if we are not, then 'do' next step.
            if self.facing != face:
                self.facing = face
                return DIR_TO_ACTION[face]
            return A['do']
        if move is not None:
            self.facing = next(d for d, a in DIR_TO_ACTION.items() if a == move)
            return move
        return None

    def _goto_and_place(self, sem, pos, ids, place_action):
        targets = find(sem, ids)
        if not targets:
            return None
        move, face = bfs_step(sem, pos, targets)
        if face is not None:
            if self.facing != face:
                self.facing = face
                return DIR_TO_ACTION[face]
            return place_action
        if move is not None:
            self.facing = next(d for d, a in DIR_TO_ACTION.items() if a == move)
            return move
        return None

    def _make(self, sem, pos, inv, item, action, needs=None):
        """Craft `item`, fetching materials and a table as needed.

        The subtle part is 'nearby a table'. crafter/objects.py checks a 9x7
        AREA around the player, not the facing tile, so once a table is within
        range we must craft immediately and NOT keep walking toward it. An
        earlier version of this method routed to the table every step and the
        facing logic flipped direction each time, leaving the expert
        oscillating left-right forever with the materials already in hand.
        """
        needs = needs or {'wood': 1}

        # Is a table already in crafting range? If so and we have the
        # materials, craft NOW -- do not walk anywhere.
        table_in_range = near(sem, pos, {MAT['table']}, TABLE_RADIUS)
        if table_in_range and all(inv[k] >= v for k, v in needs.items()):
            return action

        # Work out everything still missing, treating "a table within reach"
        # as just another requirement costing 2 wood. Without this the expert
        # sat on wood=1, stone=4 forever: it had the craft materials, so it
        # never went for wood, but 1 wood is not enough to place a table.
        wood_needed = needs.get('wood', 0) + (0 if table_in_range else 2)
        if inv['wood'] < wood_needed:
            act = self._goto_and_do(sem, pos, {MAT['tree']})
            if act is not None:
                return act
        for k in ('stone', 'coal', 'iron'):
            if needs.get(k, 0) > inv[k]:
                mat = {'stone': MAT['stone'], 'coal': MAT['coal'],
                       'iron': MAT['iron']}[k]
                act = self._goto_and_do(sem, pos, {mat})
                if act is not None:
                    return act

        # Materials in hand but no table in range. Two options, cheapest first:
        if not table_in_range:
            # (a) Place a new table, if we can spare 2 wood beyond the craft.
            if inv['wood'] >= needs.get('wood', 0) + 2:
                act = self._goto_and_place(sem, pos, WALKABLE, A['place_table'])
                if act is not None:
                    return act
            # (b) Walk to a table that already exists. Dropping this branch is
            # what broke make_stone_pickaxe: the expert would sit on wood=1,
            # stone=1 with two tables on the map, unable to afford a third and
            # unwilling to walk to either.
            act = self._goto_and_do(sem, pos, {MAT['table']})
            if act is not None and act != A['do']:
                return act          # travelling; never 'do' on a table
            if act == A['do']:
                return action       # adjacent and facing it: craft
            # (c) No table anywhere and not enough wood: go chop.
            act = self._goto_and_do(sem, pos, {MAT['tree']})
            if act is not None:
                return act
        else:
            return action
        return self._safe_random_move(sem, pos)

    def _make_iron(self, sem, pos, inv, action):
        if inv['wood'] >= 1 and inv['coal'] >= 1 and inv['iron'] >= 1:
            has_table = near(sem, pos, {MAT['table']}, TABLE_RADIUS)
            has_furnace = near(sem, pos, {MAT['furnace']}, TABLE_RADIUS)
            if has_table and has_furnace:
                return action
            # Iron tools need BOTH a table and a furnace in range. Building
            # them where we stand is far more reliable than finding a spot
            # that already has both, so carry the materials and set up shop.
            if not has_table and inv['wood'] >= 3:
                act = self._goto_and_place(sem, pos, WALKABLE, A['place_table'])
                if act is not None:
                    return act
            if not has_furnace and inv['stone'] >= 4:
                act = self._goto_and_place(sem, pos, WALKABLE, A['place_furnace'])
                if act is not None:
                    return act
            if not has_table and inv['wood'] < 3:
                act = self._goto_and_do(sem, pos, {MAT['tree']})
                if act is not None:
                    return act
            if not has_furnace and inv['stone'] < 4:
                act = self._goto_and_do(sem, pos, {MAT['stone']})
                if act is not None:
                    return act
        for item, mat in (('wood', MAT['tree']), ('coal', MAT['coal']),
                          ('iron', MAT['iron'])):
            if inv[item] < 1:
                act = self._goto_and_do(sem, pos, {mat})
                if act is not None:
                    return act
        return self._safe_random_move(sem, pos)


def play_episode(seed, length):
    """Play one episode, buffering the transitions in memory. No Recorder.

    Returns (achievements, steps, transitions) where transitions is the list
    of per-step dicts that write_npz() needs.

    WHY WE BUFFER INSTEAD OF SCREENING TWICE
    ----------------------------------------
    The obvious design is "play once to score the seed, then replay the good
    seeds with crafter.Recorder attached". That design is BROKEN on Crafter and
    it took a determinism test to find out:

      crafter/engine.py:209 draws from `self._world.random` to render the
      night-time darkness overlay -- and that is the SAME numpy RandomState
      that drives zombie and skeleton AI (crafter/objects.py:298 etc).

    So every extra render call silently advances the world's randomness.
    Attaching the Recorder renders extra frames, the enemies then behave
    differently, and the replayed episode is NOT the episode you screened.
    Measured on seed 1000: screening gave 11 achievements over 154 steps, the
    "same" recorded run gave 5 over 64. The --require filter was therefore
    guaranteeing nothing.

    Buffering one episode is cheap: a 300-step episode of 64x64x3 uint8 frames
    is about 3.7 MB. Note this is NOT the same as enabling Recorder's
    save_video, which buffers 512x512 frames and would be ~7.8 GB for a full
    10,000-step episode.
    """
    env = crafter.Env(area=(64, 64), view=(9, 9), size=(64, 64),
                      reward=True, length=length, seed=seed)
    obs = env.reset()
    transitions = [{'image': obs}]
    expert = ScriptedExpert(np.random.default_rng(seed))
    done, steps, info = False, 0, None

    while not done and steps < length:
        action = A['noop'] if info is None else expert.act(info)
        obs, reward, done, info = env.step(int(action))
        steps += 1
        # Same field layout as crafter/recorder.py:122-137, so the .npz we
        # write is interchangeable with one produced by run_gui.py --record.
        tr = {'image': obs, 'action': int(action), 'reward': float(reward),
              'done': bool(done)}
        for key, value in info.items():
            if key in ('inventory', 'achievements'):
                continue
            tr[key] = value
        for key, value in info['achievements'].items():
            tr[f'achievement_{key}'] = value
        for key, value in info['inventory'].items():
            tr[f'ainventory_{key}'] = value
        transitions.append(tr)

    return info['achievements'].copy(), steps, transitions


def write_npz(transitions, achievements, steps, outdir):
    """Write one buffered episode in crafter.Recorder's exact .npz format.

    Mirrors crafter/recorder.py:143-152, including its filename convention
    (recorder.py:184) so the files sort by achievement count.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Fill zeros for keys missing at the first timestep, as the Recorder does.
    for key, value in transitions[1].items():
        if key not in transitions[0]:
            transitions[0][key] = np.zeros_like(value)
    episode = {k: np.array([t[k] for t in transitions])
               for k in transitions[0]}
    n_unlocked = sum(int(v >= 1) for v in achievements.values())
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    name = f'{stamp}-ach{n_unlocked}-len{steps}.npz'
    np.savez_compressed(str(outdir / name), **episode)

    # stats.jsonl too, matching crafter/recorder.py:57-67, so the demo set can
    # be scored with the same analysis code as a training run.
    stats = {'length': steps,
             'reward': round(float(sum(t.get('reward', 0.0)
                                       for t in transitions[1:])), 1)}
    for key, value in achievements.items():
        stats[f'achievement_{key}'] = int(value)
    with (outdir / 'stats.jsonl').open('a') as f:
        f.write(json.dumps(stats) + '\n')
    return name


def record(args):
    """Screen many seeds, keep only the episodes worth demonstrating.

    This is the part that matters. The expert is good but high-variance -- the
    world is procedurally generated, so some seeds put stone and water within
    20 tiles and others put the player on a sand spit. Rather than tune the
    planner until the median seed is excellent, we screen cheaply and keep the
    tail. The contribution only ever asked for "10-20 episodes that reach the
    stone pickaxe", not for a uniformly strong player.
    """
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    screened, kept = [], []
    seed = args.seed
    attempts = 0
    print(f'screening seeds for episodes with >= {args.min_achievements} '
          f'achievements (want {args.episodes})...')
    if args.require:
        print(f'  and requiring: {", ".join(args.require)}')
    while len(kept) < args.episodes and attempts < args.max_attempts:
        # One pass: play, score, and keep the frames. The episode we write is
        # BY CONSTRUCTION the episode we scored -- see play_episode's docstring
        # for why a screen-then-replay design silently fails here.
        ach, steps, transitions = play_episode(seed, args.length)
        n = sum(v > 0 for v in ach.values())
        unlocked = {k for k, v in ach.items() if v > 0}
        screened.append(n)
        good = (n >= args.min_achievements
                and all(r in unlocked for r in args.require))
        if good:
            name = write_npz(transitions, ach, steps, outdir)
            kept.append((seed, n, steps, sorted(unlocked)))
            print(f'  KEEP seed {seed:>5}  ach {n:>2}/22  len {steps:>5}  '
                  f'({len(kept)}/{args.episodes})  {name}')
        elif args.verbose:
            print(f'  skip seed {seed:>5}  ach {n:>2}/22  len {steps:>5}')
        del transitions          # free the frames before the next episode
        seed += 1
        attempts += 1

    print(f'\nscreened {attempts} seeds, kept {len(kept)}, wrote to {outdir}/')
    if screened:
        print(f'screened achievement counts: mean {np.mean(screened):.1f}  '
              f'max {max(screened)}')
    if not kept:
        print('nothing kept -- lower --min-achievements or raise --max-attempts')
        return

    ach_counts = collections.Counter()
    for _, _, _, depth in kept:
        ach_counts.update(depth)
    print(f'\nachievement coverage across the {len(kept)} KEPT demos:')
    for name in crafter.constants.achievements:
        c = ach_counts[name]
        print(f'  {name:<22} {c:>3}/{len(kept)}  {"#" * c}')
    missing = [n for n in crafter.constants.achievements if ach_counts[n] == 0]
    if missing:
        print(f'\nnever demonstrated: {", ".join(missing)}')
        print('These stay at zero in the buffer, so the agent must find them '
              'alone. That is the honest limit of this demo set and belongs '
              'on the slide.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=20,
                   help='how many GOOD episodes to keep')
    p.add_argument('--outdir', type=str, default='demos')
    p.add_argument('--seed', type=int, default=1000,
                   help='first seed to screen; screening walks upward')
    p.add_argument('--length', type=int, default=10000)
    p.add_argument('--min-achievements', type=int, default=12,
                   help='keep an episode only if it unlocks at least this many')
    p.add_argument('--require', nargs='*', default=['make_stone_pickaxe'],
                   help='achievement names an episode MUST contain to be kept')
    p.add_argument('--max-attempts', type=int, default=400)
    p.add_argument('--verbose', action='store_true')
    record(p.parse_args())


if __name__ == '__main__':
    main()
