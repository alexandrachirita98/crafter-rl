"""Stage 18 -- stage 14's extended mask plus stage 04's n-step returns.

Both wins at 1M are single-flag changes on the same stage 03 base, and they
act on different parts of the update: the mask narrows the action set, n-step
carries reward back further per update. Stacking them is the obvious next
question and needs no new learning rule -- stage 14's agent IS DoubleDQN, and
stage 04's is DoubleDQN with a different buffer and _update, so taking
stage 04's agent and stage 14's mask flags composes them exactly.

The reason it is worth running: --mask-skip-grass is what finally opened the
long rewardless gap n-step is for. It took collect_stone 3.1 -> 28.9 and
make_wood_pickaxe 16.7 -> 47.4, so the 30-60 steps between holding a pickaxe
and standing on stone now happen often enough to be worth propagating.

Run:
    python stage_18_extmask_nstep/train.py --test-stack
    python stage_18_extmask_nstep/train.py --total-steps 200000 --seed 1 \
        --num-envs 16 --mask --mask-extended --mask-skip-grass --n-step 3
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS
from stage_02_dqn.train import get_estimator, main as stage_02_main
from stage_04_nstep.train import (NStepDoubleDQN, NStepReplayMemory,
                                  test_nstep, test_nstep_interleaved)
from stage_14_extended_mask.train import test_extmask

STAGE_ID = 'stage_18_extmask_nstep'


def extra_args(p):
    # Stage 02's main reads these three off args via getattr, so declaring
    # them here is all the wiring the mask needs.
    p.add_argument('--mask-extended', action='store_true',
                   help='add the do/sleep/movement rules to the mask. '
                        'Requires --mask; without it nothing is masked at all')
    p.add_argument('--mask-skip-grass', action='store_true',
                   help='also forbid `do` at grass. A behavioural prior, not '
                        'a legality rule -- report it as its own arm')
    p.add_argument('--n-step', type=int, default=3,
                   help='rewards summed before bootstrapping. 3 won the 200k '
                        'screen against 5; stage 04 defaults to 10')
    p.add_argument('--test-stack', action='store_true',
                   help='run both parents\' self-tests and exit')


def build(args, capacity):
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = NStepReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {'n_step': args.n_step}


def test_stack():
    """Neither parent's contract may change by being composed.

    There is no new rule to test: the mask is stage 14's and the window logic
    is stage 04's. What can break is the wiring -- a renamed flag or a buffer
    that no longer matches _update's unpacking -- so run both suites and
    check the agent really is the n-step one.
    """
    rc = test_nstep() or test_nstep_interleaved()
    ok = (rc == 0) and test_extmask()

    # The flags must survive parse_args, and the buffer must be the 7-tuple
    # one: a plain ReplayMemory here would feed _update 6 values and crash
    # only once training starts.
    from stage_02_dqn.train import parse_args
    args = parse_args(['--mask', '--mask-extended', '--mask-skip-grass',
                       '--n-step', '5'], extra_args=extra_args)
    if not (args.mask and args.mask_extended and args.mask_skip_grass):
        print('FAIL mask flags did not survive parse_args'); ok = False
    if args.n_step != 5:
        print('FAIL --n-step did not survive parse_args'); ok = False
    _, buf, kw = build(args, capacity=64)
    if not isinstance(buf, NStepReplayMemory):
        print('FAIL build must give the n-step buffer'); ok = False
    if kw != {'n_step': 5}:
        print(f'FAIL build must pass n_step through, got {kw}'); ok = False

    print('\nstack wiring:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-stack' in (argv if argv is not None else sys.argv[1:]):
        return test_stack()
    return stage_02_main(argv, agent_cls=NStepDoubleDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
