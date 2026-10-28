"""Stage 14 -- the extended action mask (docs/beyond-c1-c2-c3.md §3, #3).

common/masking.py only masked actions 7-16, the place_* and make_*. Actions
0-6 -- noop, the four moves, `do`, `sleep` -- were always legal. This stage
adds three rules behind --mask-extended:

  1. `do` only when the faced tile is collectable and the tool is held.
  2. `sleep` only below full energy.
  3. a move only when it is not both blocked and already faced.

Rule 3 needs the facing check: a blocked move turns the player instead of
moving, which is how the agent aims, so masking every blocked move would
leave it unable to line up a `do`. Crafter's info has no facing field;
common/env.py adds info['player_facing'].

--mask-skip-grass also forbids `do` at grass. Grass is collectable (sapling
at p=0.1), so that one can forbid an action that would have worked -- it is a
separate arm and reported separately.

The mask reads inventory, the semantic map and facing, none of which is in
the image. Compare masked runs only against our own masked baseline.

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
    """Stage 03's net and buffer, so --mask-extended is the only difference."""
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {}


# ------------------------------------------------------------- self-test


def test_extmask():
    """The three rules on hand-built info dicts.

    tests/test_masking_extended.py is the real suite; it replays every masked
    action in a deepcopy of the env. This is the smoke check run.sh wires up.
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

    # -- rule 3: facing away, a blocked move still turns us, so stays legal.
    info['player_facing'] = (0, -1)
    if legal_mask(info, extended=True)[4] != 1.0:
        print('FAIL blocked move must stay legal when not already facing it')
        ok = False
    info['player_facing'] = (0, +1)
    if legal_mask(info, extended=True)[4] != 0.0:
        print('FAIL move into a faced unwalkable tile is a no-op'); ok = False

    # -- grass is collectable, so only skip_grass may mask it.
    info['semantic'][x, y + 1] = GRASS_ID
    if legal_mask(info, extended=True)[DO] != 1.0:
        print('FAIL grass is collectable: plain extended must leave do legal')
        ok = False
    if legal_mask(info, extended=True, skip_grass=True)[DO] != 0.0:
        print('FAIL --mask-skip-grass should forbid do at grass'); ok = False

    # -- without facing, the facing rules are skipped rather than guessed.
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
