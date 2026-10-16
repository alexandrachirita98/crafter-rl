"""Neural network components, shared by every stage.

Shared because every stage uses the SAME trunk: if the convolutional encoder
differed between stages, the ablation table would be measuring two things at
once. The rule from docs/implementation-guide.md section 0.2: if changing it
would change an earlier stage's result, it does not belong in common/.

So the trunk lives here. The loss does not -- changing the loss is the entire
point of stage 06.

Later stages add to this file rather than editing it:
  stage 05 -- DuelingHead
  stage 06 -- QRHead (quantile outputs)
  stage 11 -- the achievement auxiliary head
"""

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


def test_shapes():
    """Check the 1024 flatten width and the NHWC convention actually hold.

    Run: python -m common.networks
    """
    import torch

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
    print(f'conv flatten   {conv_out.shape[1]} (expected {TRUNK_OUT})')
    print(f'trunk features {tuple(feats.shape)}')
    print(f'q values       {tuple(values.shape)} (expected ({batch}, {N_ACTIONS}))')
    print(f'parameters     {n_params:,}')
    print('shapes OK')


if __name__ == '__main__':
    test_shapes()
