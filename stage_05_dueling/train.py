"""Stage 05 -- dueling head (lab notebook section 8 bonus task).

Wang et al. 2016, https://arxiv.org/abs/1511.06581

Split "how good is this situation?" from "does this button matter?". Most
Crafter states are walking across grass, where all 17 actions are nearly
equivalent, so learning V(s) once beats learning 17 near-identical Q-values.

Only the network changes; the agent is stage 04's NStepDoubleDQN. The trunk
comes out of get_estimator, so the ablation row has one moving part.

Run:
    python stage_05_dueling/train.py --test-dueling
    python stage_05_dueling/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (DEVICE, N_ACTIONS, get_estimator,
                                main as stage_02_main)
from stage_04_nstep.train import (NStepDoubleDQN, build as build_04,
                                  extra_args as extra_04)

STAGE_ID = 'stage_05_dueling'


class DuelingNet(nn.Module):
    """get_estimator's trunk, then two streams: V(s) and A(s, a)."""

    def __init__(self, action_num, lin_size=64, stride=1):
        super().__init__()
        plain = get_estimator(action_num, lin_size=lin_size, stride=stride)
        self.trunk = plain[0]                 # ByteToFloat, three convs, View
        flat = plain[1].in_features           # the probed flatten width
        self.value = nn.Sequential(
            nn.Linear(flat, lin_size), nn.ReLU(inplace=True),
            nn.Linear(lin_size, 1))
        self.adv = nn.Sequential(
            nn.Linear(flat, lin_size), nn.ReLU(inplace=True),
            nn.Linear(lin_size, action_num))

    def forward(self, x):
        f = self.trunk(x)
        v, a = self.value(f), self.adv(f)
        # Subtract the mean advantage, or V and A drift together
        # unidentifiably. keepdim=True: without it the broadcast is (B, B)
        # and still runs.
        return v + a - a.mean(1, keepdim=True)


def extra_args(p):
    extra_04(p)
    p.add_argument('--test-dueling', action='store_true',
                   help='check mean_a Q(s,a) == V(s) and exit')


def build(args, capacity):
    _, buffer, extra = build_04(args, capacity)
    net = DuelingNet(N_ACTIONS, lin_size=args.lin_size,
                     stride=args.stride).to(DEVICE)
    return net, buffer, extra


# ------------------------------------------------------------- self-test


def test_dueling(batch=8, lin_size=16, stride=2):
    """mean over actions of Q must equal V(s) exactly.

    Fails if the `- a.mean(...)` is missing or keepdim is dropped, neither of
    which raises on its own.
    """
    net = DuelingNet(N_ACTIONS, lin_size=lin_size, stride=stride).to(DEVICE)
    x = torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8,
                      device=DEVICE)
    with torch.no_grad():
        q = net(x)
        v = net.value(net.trunk(x)).squeeze(1)

    shape_ok = tuple(q.shape) == (batch, N_ACTIONS)
    gap = (q.mean(1) - v).abs().max().item()
    mean_ok = gap < 1e-5
    print(f'q.shape       {tuple(q.shape)}  want {(batch, N_ACTIONS)}  '
          f'{"ok" if shape_ok else "WRONG"}')
    print(f'|mean_a Q - V| {gap:.2e}  want < 1e-5  '
          f'{"ok" if mean_ok else "WRONG"}')
    print('\nPASS' if shape_ok and mean_ok else '\nFAIL')
    return 0 if shape_ok and mean_ok else 1


def main(argv=None):
    if '--test-dueling' in (argv if argv is not None else sys.argv[1:]):
        return test_dueling()
    return stage_02_main(argv, agent_cls=NStepDoubleDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
