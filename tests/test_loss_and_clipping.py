"""The loss fix must bound the gradient, and must stay switchable.

Two things are being pinned. First, that Huber's gradient saturates where
MSE's grows with the error -- Crafter pays +1 after hundreds of zeros, so a
TD error of 3-10 is the normal case on an achievement, not the tail. Second,
that --loss mse --clip-grad 0 still reaches the old arithmetic, because every
score already in */results/ was produced by it.

The gradient numbers are the reason the fix exists; see
docs/colleague-bug-review.md section 2.
"""

import numpy as np
import pytest
import torch

from common.env import N_ACTIONS
import stage_02_dqn.train as s02


def _agent(loss_kind='huber', clip_grad=10.0, warmup=8):
    net = s02.get_estimator(N_ACTIONS, lin_size=16, stride=2).to(s02.DEVICE)
    return s02.DQN(net, s02.ReplayMemory(size=200, batch_size=4),
                   torch.optim.Adam(net.parameters(), lr=1e-4),
                   s02.get_epsilon_schedule(start=1.0, end=0.05, steps=50),
                   N_ACTIONS, gamma=0.99, warmup_steps=warmup, update_steps=1,
                   update_target_steps=50, num_envs=1,
                   loss_kind=loss_kind, clip_grad=clip_grad)


@pytest.mark.parametrize('error', [0.05, 0.5, 3.0, 10.0])
def test_huber_gradient_never_exceeds_mse(error):
    """Equal at small errors, strictly smaller once the error passes 1."""
    qsa = torch.zeros(8, 1, requires_grad=True)
    target = torch.full((8, 1), float(error))

    grads = {}
    for kind in ('mse', 'huber'):
        agent = _agent(loss_kind=kind)
        q = qsa.clone().detach().requires_grad_(True)
        agent._td_loss(q, target).backward()
        grads[kind] = q.grad.abs().max().item()

    assert grads['huber'] <= grads['mse'] + 1e-9
    if error > 1.0:
        # the whole point: past the kink MSE keeps scaling and Huber does not
        assert grads['huber'] < grads['mse']


def test_huber_gradient_is_bounded_but_mse_is_not():
    """MSE's gradient grows with the error; Huber's flattens at the kink."""
    target_small = torch.full((8, 1), 1.0)
    target_big = torch.full((8, 1), 20.0)

    def grad(kind, target):
        agent = _agent(loss_kind=kind)
        q = torch.zeros(8, 1, requires_grad=True)
        agent._td_loss(q, target).backward()
        return q.grad.abs().max().item()

    assert grad('mse', target_big) > 10 * grad('mse', target_small)
    assert grad('huber', target_big) == pytest.approx(
        grad('huber', target_small), rel=1e-6)


def test_clip_grad_caps_the_update_norm():
    """A deliberately huge loss must leave the net at exactly the clip norm."""
    agent = _agent(clip_grad=1.0)
    states = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8,
                           device=s02.DEVICE)
    # 1e4 is far past anything Crafter produces; the point is that the clip
    # does not care how big the input was.
    loss = (agent._estimator(states) - 1e4).pow(2).mean()
    agent._backward(loss)
    total = torch.cat([p.grad.flatten()
                       for p in agent._estimator.parameters()]).norm()
    assert total.item() == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_zero_leaves_the_gradient_alone():
    """--clip-grad 0 is the pre-fix behaviour and must not clip."""
    agent = _agent(clip_grad=0.0)
    states = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8,
                           device=s02.DEVICE)
    loss = (agent._estimator(states) - 1e4).pow(2).mean()
    agent._backward(loss)
    total = torch.cat([p.grad.flatten()
                       for p in agent._estimator.parameters()]).norm()
    assert total.item() > 1.0


def test_mse_flag_reaches_the_old_arithmetic():
    """--loss mse must be exactly (qsa - target)^2, not an approximation."""
    agent = _agent(loss_kind='mse')
    qsa = torch.randn(16, 1)
    target = torch.randn(16, 1)
    assert agent._td_loss(qsa, target).item() == pytest.approx(
        (qsa - target).pow(2).mean().item(), rel=1e-12)
