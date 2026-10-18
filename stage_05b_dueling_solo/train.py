"""Stage 05b -- the dueling head alone, on stage 03.

Stage 05 measures dueling on top of n-step returns; this row answers "dueling
vs plain Double DQN". Only the network differs: stage 03's DoubleDQN, stage
02's one-step buffer, DuelingNet imported from stage 05.

Run:
    python stage_05b_dueling_solo/train.py --test-solo
"""

import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (DEVICE, N_ACTIONS, ReplayMemory,
                                main as stage_02_main)
from stage_03_double.train import DoubleDQN
from stage_05_dueling.train import DuelingNet

STAGE_ID = 'stage_05b_dueling_solo'

# The agent under test, named once so main() and the self-test cannot drift.
AGENT_CLS = DoubleDQN


def build(args, capacity):
    """Stage 02's one-step buffer, stage 05's net. No n-step, no extras."""
    net = DuelingNet(N_ACTIONS, lin_size=args.lin_size,
                     stride=args.stride).to(DEVICE)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {}


def extra_args(p):
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation holds, then exit')


# ------------------------------------------------------------- self-test


def test_solo():
    """The point of this file is what it does not inherit.

    Checks the agent is stage 03's class with no n-step machinery, the buffer
    pushes five-tuples, and the net is still the dueling one.
    """
    import argparse
    from stage_04_nstep.train import NStepDoubleDQN

    args = argparse.Namespace(lin_size=16, stride=2, batch_size=8)
    net, buffer, extra = build(args, 64)

    # AGENT_CLS is what main() passes; assert it carries no n-step machinery
    is_double = (AGENT_CLS is DoubleDQN
                 and not issubclass(AGENT_CLS, NStepDoubleDQN)
                 and not hasattr(AGENT_CLS, '_emit'))
    plain_buf = type(buffer).__name__ == 'ReplayMemory'
    no_extra = extra == {}
    dueling = type(net).__name__ == 'DuelingNet'

    x = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8, device=DEVICE)
    with torch.no_grad():
        q = net(x)
        v = net.value(net.trunk(x)).squeeze(1)
    head_ok = tuple(q.shape) == (4, N_ACTIONS) and (
        (q.mean(1) - v).abs().max().item() < 1e-5)

    print(f'agent is DoubleDQN, not n-step   {is_double}   want True')
    print(f'buffer is one-step ReplayMemory  {plain_buf}   want True')
    print(f'no extra agent kwargs            {no_extra}   want True')
    print(f'net is the dueling head          {dueling}   want True')
    print(f'mean_a Q == V, output (4, 17)    {head_ok}   want True')
    ok = is_double and plain_buf and no_extra and dueling and head_ok
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    return stage_02_main(argv, agent_cls=AGENT_CLS, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
