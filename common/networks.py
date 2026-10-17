"""Neural network components, shared by every stage.

Shared because every stage uses the SAME trunk: if the convolutional encoder
differed between stages, the ablation table would be measuring two things at
once. The rule: if changing something would change an earlier stage's result,
it does not belong in common/.

So the trunk lives here. The loss does not -- changing the loss is the entire
point of stage 06.

Later stages add to this file rather than editing it:
  stage 11 -- the achievement auxiliary head
"""

import torch
import torch.nn as nn

from common.env import N_ACTIONS, OBS_SHAPE

# Flatten width of the NatureCNN conv stack on a 64x64 input, verified by
# test_shapes() below and asserted at construction time.
TRUNK_OUT = 1024


class NatureCNN(nn.Module):
    """Mnih et al. 2015 trunk (Nature 518:529-533), on Crafter's 64x64x3 RGB.

    The Atari original takes 84x84 greyscale stacks; the three conv layers are
    unchanged and the flatten width happens to come out at 1024 for a 64x64x3
    input, verified below.

    TWO THINGS HAPPEN IN forward() THAT ARE EASY TO FORGET, AND NEITHER CRASHES.

    1. uint8 -> float and /255. The replay buffer stores uint8 because float32
       would make it 4x bigger for zero benefit (see common/replay.py). The
       conversion therefore belongs here, on the GPU, not in the buffer.
       Skip the /255 and the inputs are 0..255, which saturates the ReLUs and
       the agent never learns.

    2. permute NHWC -> NCHW. Crafter hands back (H, W, C); torch convolutions
       want (C, H, W). Skipping it does not raise -- it convolves the wrong
       axes and silently learns nothing.

    IF THE LOSS IS FLAT, CHECK THOSE TWO FIRST.
    """

    def __init__(self, in_channels=OBS_SHAPE[2], features_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),  # -> 32x15x15
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),           # -> 64x6x6
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),           # -> 64x4x4
            nn.Flatten(),                                        # -> 1024
        )
        self.fc = nn.Sequential(nn.Linear(TRUNK_OUT, features_dim), nn.ReLU())
        self.features_dim = features_dim

    def forward(self, x):
        x = x.float() / 255.0            # see note 1
        x = x.permute(0, 3, 1, 2)        # see note 2: NHWC -> NCHW
        return self.fc(self.conv(x))


class QNetwork(nn.Module):
    """NatureCNN trunk plus one linear head: 17 action-values.

    Output row i is Q(s, a_i), the network's estimate of the total discounted
    future reward from pressing button i in state s and then playing well.
    To act: argmax. To learn: regress onto the Bellman target (stage 02).
    """

    def __init__(self, n_actions=N_ACTIONS, features_dim=512):
        super().__init__()
        self.trunk = NatureCNN(features_dim=features_dim)
        self.head = nn.Linear(features_dim, n_actions)

    def forward(self, obs):
        """obs: uint8 tensor (batch, 64, 64, 3) -> float (batch, n_actions)."""
        return self.head(self.trunk(obs))


class DuelingQNetwork(nn.Module):
    """Stage 05. Same trunk, but the head splits into V(s) and A(s, a).

        Wang et al. 2016, "Dueling Network Architectures for Deep
        Reinforcement Learning", https://arxiv.org/abs/1511.06581

    WHY IT SHOULD HELP ON CRAFTER. QNetwork's single linear head has to learn
    17 separate numbers for every state. But most Crafter states are "walking
    across grass", where all 17 actions are worth very nearly the same thing:
    the agent's position after one step barely matters. A plain head still
    spends 17 outputs re-deriving that one shared value from scratch, and only
    the action actually taken gets a gradient, so each visit teaches it about
    one of the 17.

    The split says the quantity worth learning is the state's value, and the
    per-action numbers are small corrections on top of it:

        Q(s, a) = V(s) + A(s, a)

    V is a single output, so EVERY transition trains it regardless of which
    button was pressed. That is where the sample-efficiency comes from.

    THE IDENTIFIABILITY PROBLEM, AND THE ONE LINE THAT FIXES IT. Written as
    V + A the decomposition is not unique: add 100 to V, subtract 100 from
    every A, and Q is unchanged. Nothing in the Bellman loss prefers one
    split over the other, so V and A drift together, unbounded, and the whole
    point of the architecture is lost.

    Force a unique answer by making the advantages sum to zero (paper eq. 9):

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    Now V carries the state's value and A carries only the deviations. The
    paper also offers a max-based version; eq. 9 reports the mean subtraction
    as the more stable of the two, and it is what Rainbow uses.

    NOTE THE MEAN IS NOT OPTIONAL AND ITS ABSENCE DOES NOT CRASH. Drop it and
    training still runs, the loss still falls, Q just slowly detaches from any
    meaningful scale. Watch mean-Q: drifting with no corresponding return
    improvement is this bug.
    """

    def __init__(self, n_actions=N_ACTIONS, features_dim=512):
        super().__init__()
        self.trunk = NatureCNN(features_dim=features_dim)
        self.n_actions = n_actions
        # Two hidden streams, each the same width as the plain head's input,
        # per the paper. This is the one place stage 05 adds parameters over
        # stage 03 -- the trunk is untouched, so the ablation stays clean.
        self.value = nn.Sequential(
            nn.Linear(features_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, 1))
        self.adv = nn.Sequential(
            nn.Linear(features_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, n_actions))

    def forward(self, obs):
        """obs: uint8 tensor (batch, 64, 64, 3) -> float (batch, n_actions)."""
        f = self.trunk(obs)
        v = self.value(f)                           # (B, 1), broadcasts
        a = self.adv(f)                             # (B, n_actions)
        # keepdim=True so the mean broadcasts back over the action axis. With
        # keepdim=False this silently becomes (B, B) and trains on garbage.
        return v + a - a.mean(dim=1, keepdim=True)


class QRDuelingQNetwork(nn.Module):
    """Stage 06. The dueling head, but every output is N quantiles instead of
    one number.

        Dabney, Rowland, Bellemare & Munos 2018, "Distributional
        Reinforcement Learning with Quantile Regression",
        https://arxiv.org/abs/1710.10044, section 3 and appendix A

    WHAT CHANGES. DuelingQNetwork emits one scalar per action: the EXPECTED
    return. This emits N numbers per action describing the whole SPREAD of
    returns -- "this action usually gives 3 but sometimes 12". The agent still
    acts on the mean, so the policy is unchanged in form; what changes is how
    much the network is asked to predict per sample.

    WHY THAT HELPS SO MUCH ON A SMALL BUDGET. One transition used to supply one
    regression target. It now supplies N of them, each pinned to a different
    part of the distribution, so the trunk gets N times the gradient signal
    from the same 1M environment steps. EDE -- the only DQN-family agent that
    does well on Crafter, at 11.7% -- makes exactly this swap as its FIRST
    change to Rainbow.

    WHY QUANTILES AND NOT C51. C51 fixes a support [V_min, V_max] up front and
    projects onto it; on Crafter nobody knows the right range, and a bad guess
    silently clips the distribution. QR-DQN learns the LOCATIONS and fixes the
    PROBABILITIES instead, so there is no range to guess and no projection
    step. That is why we reject C51.

    THE SHAPE, WHICH IS THE WHOLE INTERFACE: (batch, n_actions, n_quantiles).
    Every stage-06 bug is a shape or an axis. Act on

        q_values = quantiles.mean(dim=2)      # (B, n_actions)

    The mean over the quantile axis IS the Q-value -- taking mean(dim=1) by
    mistake averages over ACTIONS, which produces a tensor of the right-looking
    shape and an agent that cannot choose.

    THE DUELING SPLIT IS UNCHANGED, APPLIED PER QUANTILE. V is (B, 1, N) and A
    is (B, n_actions, N), and the mean advantage is subtracted over the ACTION
    axis (dim=1), not the quantile axis. Subtracting over dim=2 instead would
    zero each action's mean return -- destroying exactly the information the
    agent acts on -- and it does not crash.
    """

    def __init__(self, n_actions=N_ACTIONS, n_quantiles=51, features_dim=512):
        super().__init__()
        self.trunk = NatureCNN(features_dim=features_dim)
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        # Same two streams as stage 05, each widened by a factor of
        # n_quantiles at the output. The trunk is untouched, so the 06-vs-05
        # ablation row still has one moving part: the loss and this head.
        self.value = nn.Sequential(
            nn.Linear(features_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, n_quantiles))
        self.adv = nn.Sequential(
            nn.Linear(features_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, n_actions * n_quantiles))

    def forward(self, obs):
        """obs: uint8 (batch, 64, 64, 3) -> float (batch, n_actions, n_quant)."""
        f = self.trunk(obs)
        b = f.shape[0]
        v = self.value(f).view(b, 1, self.n_quantiles)
        a = self.adv(f).view(b, self.n_actions, self.n_quantiles)
        # dim=1 is the ACTION axis. See the docstring: dim=2 also runs.
        return v + a - a.mean(dim=1, keepdim=True)

    def q_values(self, obs):
        """Expected return per action, (batch, n_actions). What acting uses."""
        return self.forward(obs).mean(dim=2)


def quantile_midpoints(n_quantiles, device=None):
    """The tau values the quantile loss must be weighted by: (2i+1)/(2N).

    MIDPOINTS, NOT EDGES. Using i/N makes tau_0 = 0, whose loss weight
    |tau - 1{u<0}| is zero for every over-estimate, so the first quantile head
    receives no gradient in that direction and drifts. This is failure mode 1
    of three in stage 06.
    """
    i = torch.arange(n_quantiles, device=device, dtype=torch.float32)
    return (2.0 * i + 1.0) / (2.0 * n_quantiles)


def test_shapes():
    """Check the 1024 flatten width and the NHWC convention actually hold.

    Run: python -m common.networks
    """
    batch = 4
    obs = torch.randint(0, 256, (batch, *OBS_SHAPE), dtype=torch.uint8)

    trunk = NatureCNN()
    feats = trunk(obs)
    assert feats.shape == (batch, 512), feats.shape

    # The number the guide asserts: conv stack flattens to exactly 1024.
    conv_out = trunk.conv(obs.float().permute(0, 3, 1, 2) / 255.0)
    assert conv_out.shape == (batch, TRUNK_OUT), conv_out.shape

    q = QNetwork()
    values = q(obs)
    assert values.shape == (batch, N_ACTIONS), values.shape

    n_params = sum(p.numel() for p in q.parameters())

    # --- stage 05: the dueling head is a drop-in for QNetwork ------------
    duel = DuelingQNetwork()
    duel_values = duel(obs)
    assert duel_values.shape == (batch, N_ACTIONS), duel_values.shape

    # The identifiability constraint, checked rather than assumed: the
    # advantage stream's contribution to Q must sum to zero across actions,
    # so Q's own mean over actions has to equal V(s) exactly. This is the
    # assertion that a missing `- a.mean(...)` or a keepdim slip fails.
    with torch.no_grad():
        f = duel.trunk(obs)
        v = duel.value(f).squeeze(1)                 # (batch,)
        a = duel.adv(f)
        centred = a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(centred.mean(dim=1), torch.zeros(batch), atol=1e-5), \
        'advantages are not zero-mean'
    assert torch.allclose(duel_values.mean(dim=1), v, atol=1e-5), \
        'mean over actions of Q should equal V(s)'

    duel_params = sum(p.numel() for p in duel.parameters())

    # --- stage 06: the same head, N quantiles wide ------------------------
    n_quant = 51
    qr = QRDuelingQNetwork(n_quantiles=n_quant)
    quantiles = qr(obs)
    assert quantiles.shape == (batch, N_ACTIONS, n_quant), quantiles.shape

    # Acting collapses the QUANTILE axis, never the action axis.
    qr_q = qr.q_values(obs)
    assert qr_q.shape == (batch, N_ACTIONS), qr_q.shape
    assert torch.allclose(qr_q, quantiles.mean(dim=2), atol=1e-6)

    # The dueling constraint again, now per quantile: the advantage stream is
    # centred over ACTIONS (dim=1), so for every single quantile index the
    # mean over actions of Q equals that quantile of V. Centring over dim=2
    # by mistake passes neither of these.
    with torch.no_grad():
        f = qr.trunk(obs)
        v_q = qr.value(f).view(batch, n_quant)                  # (B, N)
    assert torch.allclose(quantiles.mean(dim=1), v_q, atol=1e-5), \
        'mean over actions of each quantile should equal V(s) quantile'

    qr_params = sum(p.numel() for p in qr.parameters())

    # taus must be MIDPOINTS. tau_0 = 0 (the i/N mistake) gives the first
    # quantile head a zero loss weight for every over-estimate, so it never
    # learns. Failure mode 1 of three in the guide's stage 06.
    taus = quantile_midpoints(n_quant)
    assert taus.shape == (n_quant,), taus.shape
    assert abs(taus[0].item() - 1.0 / (2 * n_quant)) < 1e-6, \
        'taus look like edges i/N, not midpoints (2i+1)/(2N)'
    assert abs(taus.mean().item() - 0.5) < 1e-6

    print(f'conv flatten   {conv_out.shape[1]} (expected {TRUNK_OUT})')
    print(f'trunk features {tuple(feats.shape)}')
    print(f'q values       {tuple(values.shape)} (expected ({batch}, {N_ACTIONS}))')
    print(f'dueling values {tuple(duel_values.shape)}  mean_a Q == V: OK')
    print(f'qr quantiles   {tuple(quantiles.shape)}  '
          f'mean over dim=2 -> {tuple(qr_q.shape)}: OK')
    print(f'taus[0]        {taus[0].item():.6f} '
          f'(expected {1 / (2 * n_quant):.6f}, NOT 0.0)')
    print(f'parameters     {n_params:,} plain / {duel_params:,} dueling '
          f'(+{100 * (duel_params - n_params) / n_params:.0f}%) / '
          f'{qr_params:,} qr-dueling '
          f'(+{100 * (qr_params - duel_params) / duel_params:.0f}% over dueling)')
    print('shapes OK')


if __name__ == '__main__':
    test_shapes()
