"""Stage 14 -- the extended action mask (docs/beyond-c1-c2-c3.md §3, #3).

The champion's single biggest win is legal-action masking: at 200k with 16
envs it took the score from 3.25% to 4.56%. But common/masking.py only ever
masked actions 7-16, the four place_* and six make_*. Actions 0-6 -- noop,
the four moves, `do` and `sleep` -- were hardcoded always-legal, including
`do`, which is the single most important button in the game.

This stage finishes the job. Three rules, in common/masking.py:

  1. `do` is masked unless the tile the player FACES is something `do` acts
     on and the tool requirement is met. Stone and coal need a wood pickaxe,
     iron a stone pickaxe, diamond an iron pickaxe (data.yaml collect:).
  2. `sleep` is masked at full energy -- objects.py:115 gates on
     energy < max, so the press is a guaranteed no-op.
  3. A move is masked only when the target tile is unwalkable AND the player
     already faces it.

RULE 3 IS THE DANGEROUS ONE AND IT IS WHY THE FACING CHECK IS THERE. In
Crafter, Player._move sets facing FIRST and then tries to step
(objects.py:174-177), so a move into a tile you cannot enter TURNS you to
face it -- which is exactly how the agent lines up a `do`. Masking every
blocked move would leave it unable to aim. Masking only the moves that are
redundant as turns is provably a no-op, which is what
tests/test_masking_extended.py checks by replaying every masked action in a
deepcopy of the env.

WHAT THIS BUYS, MEASURED, NOT ASSUMED. Over 10k random steps the extended
mask takes the mean legal-action count from 7.16 to 6.49 of 17. `do` is
masked on only 3.9% of steps -- far less than the doc assumed, because the
agent faces GRASS on 84.4% of steps and `do` at grass is legal: data.yaml
gives a sapling with probability 0.1. So the strict rule is a small prune.

Hence --mask-skip-grass, a SECOND and weaker-justified arm: it also forbids
`do` at grass, taking the masked fraction to 92.6% of steps. collect_sapling
is already at 91% in the champion, so those presses buy an achievement that
is long since solved -- but it is a behavioural prior, not a legality rule,
and it is the one part that can forbid an action that would have worked. It
is reported as its own arm, never folded into the headline number.

FACING IS NOT IN CRAFTER'S info DICT. common/env.py's VecEnv adds
info['player_facing'] by reading env._player.facing -- an attribute read, so
it cannot change dynamics, verified by replaying a fixed action list with and
without the read. Patching crafter/env.py would have invalidated every
measurement taken before the change, which is that file's own rule.

PRIVILEGED STATE, SAID OUT LOUD. The mask reads inventory, the semantic map
and now facing: none of it is in the 64x64 image. Report masked numbers only
against our own masked baseline, never against the Crafter paper's
pixels-only table.

Run:
    python stage_14_extended_mask/train.py --test-extmask
    python stage_14_extended_mask/train.py --total-steps 200000 --seed 1 \
        --num-envs 16 --mask --mask-extended --seed-stride 16
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS
from common.masking import (DO, ENERGY_MAX, GRASS_ID, SLEEP, STONE_ID,
                            legal_mask)
from stage_02_dqn.train import ReplayMemory, get_estimator, main as stage_02_main
from stage_03_double.train import DoubleDQN

STAGE_ID = 'stage_14_extended_mask'


def extra_args(p):
    p.add_argument('--mask-extended', action='store_true',
                   help='add the do/sleep/movement rules to the mask. '
                        'Requires --mask; without it nothing is masked at all')
    p.add_argument('--mask-skip-grass', action='store_true',
                   help='also forbid `do` at grass. A behavioural prior, not '
                        'a legality rule -- report it as its own arm')
    p.add_argument('--test-extmask', action='store_true',
                   help='check the three rules against the engine, then exit')


def build(args, capacity):
    """Unchanged from stage 03: the contribution is in the mask, not the net.

    Deliberately identical so that --mask-extended on/off is a one-flag
    comparison with the same code path on both sides (docs §6 rule 1).
    """
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {}


# ------------------------------------------------------------- self-test


def test_extmask():
    """The three rules on hand-built info dicts, plus a no-op probe.

    tests/test_masking_extended.py is the real suite -- it replays every
    masked action in a deepcopy of the env, both directions. This is the
    smoke check run.sh wires up automatically.
    """
    import crafter

    ok = True
    env = crafter.Env(seed=1)
    env.reset()
    env.step(0)
    p = env._player
    info = {'inventory': dict(p.inventory), 'achievements': dict(p.achievements),
            'semantic': env._sem_view(), 'player_pos': np.array(p.pos),
            'player_facing': (0, 1)}
    x, y = int(info['player_pos'][0]), int(info['player_pos'][1])

    # -- rule 2: sleep at full energy does nothing.
    info['inventory']['energy'] = ENERGY_MAX
    if legal_mask(info, extended=True)[SLEEP] != 0.0:
        print('FAIL sleep should be masked at full energy'); ok = False
    info['inventory']['energy'] = ENERGY_MAX - 1
    if legal_mask(info, extended=True)[SLEEP] != 1.0:
        print('FAIL sleep should be legal below full energy'); ok = False

    # -- rule 1: `do` at stone needs a wood pickaxe.
    info['semantic'][x, y + 1] = STONE_ID
    info['inventory']['wood_pickaxe'] = 0
    if legal_mask(info, extended=True)[DO] != 0.0:
        print('FAIL do at stone without a pickaxe should be masked'); ok = False
    info['inventory']['wood_pickaxe'] = 1
    if legal_mask(info, extended=True)[DO] != 1.0:
        print('FAIL do at stone WITH a pickaxe should be legal'); ok = False

    # -- rule 3: the aiming trap. Facing away, a blocked move still turns us.
    info['player_facing'] = (0, -1)
    if legal_mask(info, extended=True)[4] != 1.0:
        print('FAIL blocked move must stay legal when not already facing it')
        ok = False
    info['player_facing'] = (0, +1)
    if legal_mask(info, extended=True)[4] != 0.0:
        print('FAIL move into a faced unwalkable tile is a no-op'); ok = False

    # -- skip_grass is opt-in and separate from legality.
    info['semantic'][x, y + 1] = GRASS_ID
    if legal_mask(info, extended=True)[DO] != 1.0:
        print('FAIL grass is collectable: plain extended must leave do legal')
        ok = False
    if legal_mask(info, extended=True, skip_grass=True)[DO] != 0.0:
        print('FAIL --mask-skip-grass should forbid do at grass'); ok = False

    # -- no facing: the facing rules must be skipped, not guessed.
    noface = dict(info)
    del noface['player_facing']
    m = legal_mask(noface, extended=True)
    if m[DO] != 1.0 or not m[1:5].all():
        print('FAIL without player_facing the facing rules must be skipped')
        ok = False

    print('extended mask self-test:', 'OK' if ok else 'FAILED')
    return ok


def main(argv=None):
    if '--test-extmask' in (argv if argv is not None else sys.argv[1:]):
        raise SystemExit(0 if test_extmask() else 1)
    return stage_02_main(argv, agent_cls=DoubleDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    main()
