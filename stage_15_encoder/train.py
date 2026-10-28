"""Stage 15 -- the encoder is the bottleneck (docs/beyond-c1-c2-c3.md §3, #1).

THE CLAIM. Nine algorithmic stages (Double, n-step, dueling, QR, Munchausen,
PER, ez-greedy, DrQ) all sat at 1.7-2.4% at 200k, while two non-algorithmic
changes -- 16 parallel envs and action masking -- moved the same agent to
4.6%. That says the binding constraint on Crafter at this budget is not the
RL algorithm. This stage tests the next non-algorithmic suspect: the conv
encoder, which nobody ever tuned.

WHAT THE CHAMPION'S ENCODER ACTUALLY IS (stage_02_dqn/train.py
get_estimator, the course lab's net, measured not guessed):

    Conv2d(3, 16, k=3, s=1) -> 62x62      3,690,097 params total
    Conv2d(16,16, k=2, s=1) -> 61x61      of which Linear(57600, 64)
    Conv2d(16,16, k=2, s=1) -> 60x60      is 3,686,464 -- 99.9%
    Flatten                 -> 57,600
    Linear(57600 -> 64)
    Linear(64 -> 17)

Two things are wrong with it at once:

  1. IT BARELY LOOKS AT THE IMAGE. Three stride-1 convs with k=3,2,2 shrink
     64x64 to 60x60. Each output pixel summarises a 5x5 input window. Crafter
     tiles are ~7px, so the net never forms a whole-tile concept, let alone
     "there is a mountain to my left".
  2. ALMOST ALL CAPACITY IS IN ONE SPATIALLY-BLIND LAYER. Linear(57600, 64)
     treats pixel (3,7) and (3,8) as unrelated inputs, and squeezes the world
     into 64 numbers before deciding anything -- position, facing, health,
     food, drink, energy, nine inventory counts and nine visible tiles.

THE EVIDENCE WE ALREADY HAD, AND WHAT IT RULES OUT. 200k_lin512_mask scored
4.39/4.32 against the champion's 4.80/4.32: a tie. So widening the bottleneck
ALONE does nothing -- which is what you would predict if the convolutions
never built spatial features for it to read. That is why this stage is about
the CONVOLUTIONS, and --lin-size is left to stage 02's flag.

    --encoder lab      the champion's net. The control arm.           3.69M
    --encoder stride2  the lab's convs with stride 2, lin 512.         413k
                       Downsamples 64 -> 31 -> 15 -> 7. ~5x faster.
    --encoder nature   NatureCNN from common/networks.py, unused        609k
                       until now. Downsamples 64 -> 15 -> 6 -> 4,
                       flatten 1024, bottleneck 512.

NatureCNN is 6x SMALLER than the champion and has an 8x WIDER bottleneck.
Both of those point the same way: the champion spends its parameters in the
one place they cannot encode geometry.

WHY stride2 IS WORTH RUNNING EVEN IF nature WINS. It is ~5x faster per step,
so on a fixed RunPod budget it buys more seeds -- and with a seed-to-seed
spread of 0.54 points on this champion, seeds are what the conclusion rests
on.

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

    THE ONE INCOMPATIBILITY, AND IT DOES NOT CRASH. common/networks.NatureCNN
    takes NHWC uint8 and does its own permute and /255 (see its docstring).
    stage 02's _to_tensor already hands us NCHW, because the lab's net starts
    with ByteToFloat and a Conv2d. Feeding NCHW to NatureCNN would permute a
    second time and convolve the wrong axes -- it would train, and learn
    nothing. So permute back here, once, explicitly.
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
        # The flags exist on stage 02; this just pins the pair that the doc
        # proposes, so the arm is one word on the command line and cannot be
        # half-applied.
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
    """Every encoder must take stage 02's NCHW uint8 and return (B, 17).

    The permute in NatureQ is the thing worth testing: get it wrong and the
    net trains silently on transposed images.
    """

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

    # The permute is load-bearing. Feeding stage 02's NCHW straight to
    # QNetwork makes it permute a second time, turning (B,3,64,64) into
    # (B,64,3,64) -- 64 channels into a 3-channel conv. Here that raises,
    # which is the good case; assert it does, so that if someone "simplifies"
    # NatureQ away the test fails loudly instead of the agent learning
    # nothing from transposed images.
    nat = make_encoder('nature', _Args())
    try:
        with torch.no_grad():
            nat.net(obs)          # the un-permuted path, i.e. the bug
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
