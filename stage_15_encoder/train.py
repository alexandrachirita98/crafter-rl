"""Stage 15 -- the conv encoder (docs/beyond-c1-c2-c3.md §3, #1).

Nine algorithmic stages all sat at 1.7-2.4% at 200k while two
non-algorithmic changes (16 envs, action masking) reached 4.6%, so this
stage tests the next non-algorithmic suspect: the encoder, never tuned.

stage 02's get_estimator is three stride-1 convs (64x64 -> 60x60, a 5x5
receptive field) followed by Linear(57600, 64), which holds 99.9% of its
3.69M parameters and is spatially blind. Crafter tiles are ~7px, so the
convs never form a whole-tile feature.

    --encoder lab      stage 02's net, the control arm.              3.69M
    --encoder stride2  the same convs at stride 2, lin 512.           413k
    --encoder nature   NatureCNN from common/networks.py.             609k

200k_lin512_mask already tied the champion, so widening the bottleneck alone
does nothing; stride2 and nature are the arms that change the convolutions.
stride2 also runs ~5x faster, which buys more seeds on a fixed budget.

Run:
    python stage_15_encoder/train.py --test-encoder
    python stage_15_encoder/train.py --total-steps 200000 --seed 1 \
        --num-envs 16 --mask --seed-stride 16 --encoder nature
"""

import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, OBS_SHAPE
from common.networks import QNetwork
from stage_02_dqn.train import (DEVICE, ReplayMemory, get_estimator,
                                main as stage_02_main)
from stage_03_double.train import DoubleDQN

STAGE_ID = 'stage_15_encoder'


class NatureQ(torch.nn.Module):
    """common/networks.QNetwork behind stage 02's tensor convention.

    NatureCNN expects NHWC and permutes internally; stage 02's _to_tensor
    hands out NCHW. Without the permute back it convolves the wrong axes and
    trains on transposed images without crashing.
    """

    def __init__(self, n_actions=N_ACTIONS, features_dim=512):
        super().__init__()
        self.net = QNetwork(n_actions=n_actions, features_dim=features_dim)

    def forward(self, obs):
        # NCHW (stage 02's convention) -> NHWC (what NatureCNN expects).
        return self.net(obs.permute(0, 2, 3, 1))


ENCODERS = ('lab', 'stride2', 'nature')


def make_encoder(kind, args):
    """-> an nn.Module taking NCHW uint8, returning (B, 17)."""
    if kind == 'lab':
        return get_estimator(N_ACTIONS, lin_size=args.lin_size,
                             stride=args.stride)
    if kind == 'stride2':
        # stage 02 has both flags; pinning them here keeps the arm one word.
        return get_estimator(N_ACTIONS, lin_size=512, stride=2)
    if kind == 'nature':
        return NatureQ(N_ACTIONS, features_dim=args.lin_size
                       if args.lin_size >= 512 else 512).to(DEVICE)
    raise ValueError(f'unknown encoder {kind!r}')


def extra_args(p):
    p.add_argument('--encoder', choices=ENCODERS, default='lab',
                   help="lab: the champion's net, the control arm. "
                        'stride2: the lab convs downsampling properly, '
                        '~5x faster. nature: NatureCNN from common/networks')
    p.add_argument('--test-encoder', action='store_true',
                   help='shape and parameter check on all three, then exit')


def build(args, capacity):
    net = make_encoder(args.encoder, args)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    n_params = sum(q.numel() for q in net.parameters())
    print(f'encoder {args.encoder}: {n_params:,} parameters')
    return net, buffer, {}


# ------------------------------------------------------------- self-test


def test_encoder():
    """Every encoder takes stage 02's NCHW uint8 and returns (B, 17)."""

    class _Args:
        lin_size = 64
        stride = 1

    ok = True
    batch = 4
    obs = torch.randint(0, 256, (batch, OBS_SHAPE[2], OBS_SHAPE[0],
                                 OBS_SHAPE[1]), dtype=torch.uint8).to(DEVICE)
    for kind in ENCODERS:
        net = make_encoder(kind, _Args())
        with torch.no_grad():
            out = net(obs)
        n = sum(q.numel() for q in net.parameters())
        if out.shape != (batch, N_ACTIONS):
            print(f'FAIL {kind}: got {tuple(out.shape)}, '
                  f'want {(batch, N_ACTIONS)}')
            ok = False
        if not torch.isfinite(out).all():
            print(f'FAIL {kind}: non-finite output')
            ok = False
        print(f'  {kind:<8} {tuple(out.shape)}  {n:>10,} params')

    # Feeding NCHW straight to QNetwork permutes twice, putting 64 channels
    # into a 3-channel conv. Assert it raises, so dropping NatureQ's permute
    # fails here instead of training silently on transposed images.
    nat = make_encoder('nature', _Args())
    try:
        with torch.no_grad():
            nat.net(obs)          # the un-permuted path
    except RuntimeError:
        pass                      # expected: channel mismatch
    else:
        print('FAIL nature: NCHW reached QNetwork without raising -- the '
              'permute in NatureQ is no longer load-bearing, check it')
        ok = False

    print('encoder self-test:', 'OK' if ok else 'FAILED')
    return ok


def main(argv=None):
    if '--test-encoder' in (argv if argv is not None else sys.argv[1:]):
        raise SystemExit(0 if test_encoder() else 1)
    return stage_02_main(argv, agent_cls=DoubleDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    main()
