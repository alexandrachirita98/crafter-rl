"""Variant A: coal and iron right after the stone pickaxe.

In record_demos.py the ladder is sequential and coal sits at step 8, behind the
stone sword, saplings and shelter. On seeds 5000-5039 the expert dies around
step 322 and never gets there, so collect_coal is 0/40 -- even though coal was
pathfind-reachable on 99% of steps. It is the order, not the terrain.

Coal only needs a wood pickaxe, which lands around step 50, so this variant
hoists coal and iron above everything optional. The trade is breadth for
depth: stone sword, sapling, plant and place_stone all move behind coal.

    python record_demos_coal_first.py --episodes 20 --outdir demos_coal_first
"""
import numpy as np

import crafter
from record_demos import (A, FURNACE_STONE, MAT, OBJ, SHELTER_STONE, WALKABLE,
                          ScriptedExpert, build_parser, record)


class CoalFirstExpert(ScriptedExpert):
    """Coal and iron hoisted up the ladder.

    Only _tech_tree is overridden, so any difference comes from goal order.
    """

    def _tech_tree(self, sem, pos, inv, ach):
        # 1. wood. No 3-wood float for the stone sword here -- the sword is
        # no longer the next goal, so a spare table would just cost time.
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
        # 3. wood tools
        if not ach['make_wood_pickaxe']:
            return self._make(sem, pos, inv, 'wood_pickaxe', A['make_wood_pickaxe'])
        if not ach['make_wood_sword']:
            return self._make(sem, pos, inv, 'wood_sword', A['make_wood_sword'])

        # 4. one stone, for collect_stone and the pickaxe. No bulk float
        # like the baseline's: place_stone/furnace come after coal now.
        if inv['stone'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['stone']},
                                    avoid=self.placed_stone)
            if act is not None:
                return act

        # 5. stone pickaxe -- the gate for iron.
        if not ach['make_stone_pickaxe']:
            return self._make(sem, pos, inv, 'stone_pickaxe', A['make_stone_pickaxe'],
                              needs={'wood': 1, 'stone': 1})

        # 6. coal, then iron -- the point of this variant.
        if inv['coal'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['coal']})
            if act is not None:
                return act
        if inv['iron'] < 1:
            act = self._goto_and_do(sem, pos, {MAT['iron']})
            if act is not None:
                return act

        # 7. furnace as soon as the stone is there -- not nested under
        # coal+iron. It needs only stone, and gating it on coal left 1640
        # steps holding 4 stone with no furnace placed (seeds 7000-7019).
        if not ach['place_furnace'] and self._spare_stone(inv) >= FURNACE_STONE:
            act = self._goto_and_place(sem, pos, WALKABLE, A['place_furnace'])
            if act is not None:
                return act

        # 8. iron tools, once coal and iron are in hand. No place_furnace
        # gate: _make_iron already builds whichever station is missing, and
        # gating on it sent 1290 full-recipe steps to the stone-mining branch
        # above while _make_iron ran 75 times (seeds 9000-9061).
        if inv['coal'] >= 1 and inv['iron'] >= 1:
            if not ach['make_iron_pickaxe']:
                return self._make_iron(sem, pos, inv, A['make_iron_pickaxe'])
            if not ach['make_iron_sword']:
                return self._make_iron(sem, pos, inv, A['make_iron_sword'])
        if not ach['collect_diamond'] and ach['make_iron_pickaxe']:
            act = self._goto_and_do(sem, pos, {MAT['diamond']})
            if act is not None:
                return act

        # --- baseline tail, now only once the iron branch is exhausted ---
        if ach['make_stone_pickaxe'] and not ach['make_stone_sword']:
            if inv['wood'] < 3:
                act = self._goto_and_do(sem, pos, {MAT['tree']})
                if act is not None:
                    return act
            if inv['stone'] < 1:
                act = self._goto_and_do(sem, pos, {MAT['stone']},
                                        avoid=self.placed_stone)
                if act is not None:
                    return act
            return self._make(sem, pos, inv, 'stone_sword', A['make_stone_sword'],
                              needs={'wood': 1, 'stone': 1})

        if not ach['collect_sapling']:
            act = self._goto_and_do(sem, pos, {MAT['grass']})
            if act is not None:
                return act
        if not ach['place_plant'] and inv['sapling'] >= 1:
            act = self._goto_and_place(sem, pos, {MAT['grass']}, A['place_plant'])
            if act is not None:
                return act

        # Stock up before the two stone spends; _spare_stone protects the
        # shelter reserve.
        if (not ach['place_stone'] or not ach['place_furnace']) \
                and inv['stone'] < SHELTER_STONE + 1 + FURNACE_STONE:
            act = self._goto_and_do(sem, pos, {MAT['stone']},
                                    avoid=self.placed_stone)
            if act is not None:
                return act
        if not ach['place_stone'] and self._spare_stone(inv) >= 1:
            act = self._goto_and_place(sem, pos, WALKABLE, A['place_stone'])
            if act is not None:
                return act
        if not ach['place_furnace'] and self._spare_stone(inv) >= FURNACE_STONE:
            act = self._goto_and_place(sem, pos, WALKABLE, A['place_furnace'])
            if act is not None:
                return act
        if not ach['eat_plant']:
            act = self._goto_and_do(sem, pos, {OBJ['plant']})
            if act is not None:
                return act

        if self.rng.random() < 0.3:
            return A['do']
        return self._safe_random_move(sem, pos)


def main():
    p = build_parser(description=__doc__)
    p.set_defaults(outdir='demos_coal_first')
    record(p.parse_args(), expert_cls=CoalFirstExpert)


if __name__ == '__main__':
    main()
