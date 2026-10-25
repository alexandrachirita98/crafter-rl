"""The bootstrap target must not maximise over actions the policy cannot take.

Stage 02 masked at action selection only, so the target's max ranged over
Q-values for actions masking guarantees are never executed -- and so never
corrected by experience. These tests pin the mask to the target as well, and
pin that an unmasked run is unaffected.

See docs/bugfix-implementation-and-run-order.md section 2.
"""

import numpy as np
import pytest
import torch

from common.env import N_ACTIONS, VecEnv
from common.masking import legal_masks
import stage_02_dqn.train as s02
import stage_03_double.train as s03


def _agent(cls=s02.DQN, num_envs=1, warmup=8, batch=4):
    net = s02.get_estimator(N_ACTIONS, lin_size=16, stride=2).to(s02.DEVICE)
    return cls(net, s02.ReplayMemory(size=500, batch_size=batch),
               torch.optim.Adam(net.parameters(), lr=1e-4),
               s02.get_epsilon_schedule(start=1.0, end=0.05, steps=50),
               N_ACTIONS, gamma=0.99, warmup_steps=warmup, update_steps=1,
               update_target_steps=50, num_envs=num_envs)


def test_learn_stores_the_next_state_mask():
    """The mask handed to learn() must come back out of sample()."""
    agent = _agent(batch=4)
    legal = np.zeros(N_ACTIONS, dtype=np.float32)
    legal[[0, 3, 5]] = 1.0
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(8):
        agent.learn(0, frame, 1, 0.0, frame, 1.0, legal)

    *_, sampled = agent._buffer.sample()
    assert sampled.shape == (4, N_ACTIONS)
    # every row is the mask we stored, so the target can actually use it
    assert torch.equal(sampled[0].cpu(), torch.from_numpy(legal))


def test_learn_without_a_mask_stores_all_ones():
    """An unmasked run must store a no-op mask, not None and not zeros."""
    agent = _agent(batch=4)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(8):
        agent.learn(0, frame, 1, 0.0, frame, 1.0)

    *_, sampled = agent._buffer.sample()
    assert sampled.min().item() == 1.0


@pytest.mark.parametrize('cls', [s02.DQN, s03.DoubleDQN])
def test_target_never_bootstraps_off_an_illegal_action(cls):
    """Force one action to dominate, forbid it, and check it is not used.

    The target is read back out of the gradient: with a single legal action
    the masked target must equal r + gamma * Q(s', that action) * mask, which
    is a different number from the unmasked max whenever the illegal action
    is the largest -- which is what the rigged head guarantees.
    """
    agent = _agent(cls=cls, batch=2)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    # Rig both heads so action 16 is by far the best everywhere.
    for net in (agent._estimator, agent._target_estimator):
        head = [m for m in net if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.bias[16] = 100.0
            head.bias[2] = 1.0

    legal = np.zeros(N_ACTIONS, dtype=np.float32)
    legal[2] = 1.0                      # only action 2 is legal at s'
    states = s02._to_tensor(np.stack([frame, frame]))
    actions = torch.zeros(2, 1, dtype=torch.int64, device=s02.DEVICE)
    rewards = torch.zeros(2, 1, device=s02.DEVICE)
    mask = torch.ones(2, 1, device=s02.DEVICE)
    legal_t = torch.from_numpy(np.stack([legal, legal])).to(s02.DEVICE)

    agent._update(states, actions, rewards, states, mask, legal_t)
    masked_target = agent.loss

    # the same update with nothing forbidden bootstraps off action 16
    agent2 = _agent(cls=cls, batch=2)
    for net in (agent2._estimator, agent2._target_estimator):
        head = [m for m in net if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.bias[16] = 100.0
            head.bias[2] = 1.0
    agent2._update(states, actions, rewards, states, mask,
                   torch.ones(2, N_ACTIONS, device=s02.DEVICE))
    unmasked_target = agent2.loss

    # bootstrapping off 1.0 instead of 100.0 is a far smaller loss
    assert masked_target < unmasked_target


def test_all_ones_mask_changes_nothing():
    """The masked code path on an unmasked run must be bit-identical."""
    agent = _agent(batch=2)
    frame = np.random.RandomState(0).randint(
        0, 256, (64, 64, 3)).astype(np.uint8)
    states = s02._to_tensor(np.stack([frame, frame]))
    actions = torch.zeros(2, 1, dtype=torch.int64, device=s02.DEVICE)
    rewards = torch.zeros(2, 1, device=s02.DEVICE)
    mask = torch.ones(2, 1, device=s02.DEVICE)

    with torch.no_grad():
        q_ = agent._target_estimator(states)
        plain = q_.max(1, keepdim=True)[0]
        ones = torch.ones(2, N_ACTIONS, device=s02.DEVICE)
        via_mask = s02._apply_mask(q_, ones).max(1, keepdim=True)[0]
    assert torch.equal(plain, via_mask)


def test_masked_run_end_to_end_stores_real_masks():
    """Drive a real VecEnv with --mask on and check the stored masks vary."""
    s02.reset_rng(5)
    agent = _agent(num_envs=2, warmup=12, batch=8)
    env = VecEnv(2, logdir=None, seed=5)
    states = env.reset()
    masks = None
    seen = []
    for _ in range(12):
        actions = agent.step(states, masks)
        states_, rewards, dones, infos = env.step(actions)
        masks = legal_masks(infos, dones)
        for i in range(2):
            agent.learn(i, states[i], actions[i], rewards[i], states_[i],
                        1.0, masks[i])
            seen.append(masks[i].copy())
        states = states_

    stored = np.stack(seen)
    # the point of masking: some actions are genuinely forbidden out there
    assert stored.min() == 0.0
    *_, sampled = agent._buffer.sample()
    assert sampled.shape == (8, N_ACTIONS)
