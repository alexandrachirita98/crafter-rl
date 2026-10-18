"""Variant B: grab coal and iron opportunistically, ladder untouched.

Same finding as variant A -- collect_coal is 0/40 because coal sits at step 8
of a ladder the expert dies before reaching, not because it is far away. A
reorders the ladder and pays for it in breadth; this takes the cheaper bet.

The rule: if a coal or iron tile is already adjacent and we hold the pickaxe
for it, mine it. One or two actions, no pathfinding, no detour. It sits below
the survival interrupts (a free action is not worth taking with a zombie on
you) and above the tech tree (which would never ask for coal).

Strictly cheaper than A and strictly weaker -- it only gets coal the expert
happens to walk past. Whether that is enough is the empirical question.

    python record_demos_opportunistic.py --episodes 20 --outdir demos_opportunistic
"""
import crafter
from record_demos import (A, DIR_TO_ACTION, MAT, ScriptedExpert, build_parser,
                          near, record)


class OpportunisticExpert(ScriptedExpert):
    """ScriptedExpert plus a free-action grab for adjacent coal and iron.

    _tech_tree only prepends the grab; the rest is the baseline's.
    """

    def _tech_tree(self, sem, pos, inv, ach):
        act = self._grab_adjacent(sem, pos, inv)
        if act is not None:
            return act
        return super()._tech_tree(sem, pos, inv, ach)

    def _grab_adjacent(self, sem, pos, inv):
        """Mine an adjacent coal or iron tile, if we can and it is free.

        Returns None unless the tile is within one step, so it can never pull
        the expert off its current goal. Costs one or two actions.
        """
        # Coal needs a wood pickaxe, iron a stone pickaxe (data.yaml).
        # Swinging without the tool just wastes the step.
        wants = []
        if inv['wood_pickaxe'] >= 1:
            wants.append(MAT['coal'])
        if inv['stone_pickaxe'] >= 1:
            wants.append(MAT['iron'])
        if not wants:
            return None

        for mat in wants:
            if not near(sem, pos, {mat}, 1):
                continue
            # near() includes diagonals but 'do' only reaches the faced
            # tile, so confirm an orthogonal neighbour first.
            for delta, move in DIR_TO_ACTION.items():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if not (0 <= nxt[0] < sem.shape[0]
                        and 0 <= nxt[1] < sem.shape[1]):
                    continue
                if sem[nxt] != mat:
                    continue
                if self.facing != delta:
                    # No turn action in Crafter, but the target is solid, so
                    # the move is blocked and only changes facing.
                    self.facing = delta
                    return move
                return A['do']
        return None


def main():
    p = build_parser(description=__doc__)
    p.set_defaults(outdir='demos_opportunistic')
    record(p.parse_args(), expert_cls=OpportunisticExpert)


if __name__ == '__main__':
    main()
