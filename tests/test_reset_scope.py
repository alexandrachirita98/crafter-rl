"""A "head reset" must reset a head, not the whole network.

The conv stack in get_estimator is 2,528 parameters with a 5-pixel receptive
field -- less than one 7.1-pixel Crafter tile -- so the representation lives
in the wide Linear(flat, lin_size) layers above it, not in the convs. Reset
every Linear in both streams and you re-initialise 99.97% of the net while
calling it a head reset, which is a fresh network with a warm buffer rather
than Nikishin et al.'s mechanism.

See docs/performance-issue-survey.md section 2.
"""

import torch
import torch.nn as nn

from common.env import N_ACTIONS
from stage_02_dqn.train import get_estimator
from stage_06_qrdqn.train import QRDuelingNet
from stage_10_drq_resets.train import reset_last_linear


def _fraction(scope, lin_size=64, stride=1):
    net = QRDuelingNet(N_ACTIONS, n_quantiles=51, lin_size=lin_size,
                       stride=stride)
    touched, total = reset_last_linear(net, (net.value, net.adv), scope)
    return touched / total


def test_default_scope_touches_a_small_fraction():
    """scope='last' must be a head, by any reasonable reading of the word."""
    assert _fraction('last') < 0.02


def test_all_scope_is_the_whole_network():
    """Pin what the pre-flag behaviour actually was, so the flag has meaning."""
    assert _fraction('all') > 0.99


def test_scope_last_keeps_the_wide_layer():
    """The Linear(flat, lin_size) weights must survive a reset."""
    net = QRDuelingNet(N_ACTIONS, n_quantiles=51, lin_size=64, stride=1)
    wide_before = net.value[0].weight.detach().clone()
    out_before = net.value[-1].weight.detach().clone()

    reset_last_linear(net, (net.value, net.adv), 'last')

    assert torch.equal(wide_before, net.value[0].weight.detach()), \
        'the wide layer was re-initialised; that is where the features are'
    assert not torch.equal(out_before, net.value[-1].weight.detach()), \
        'the output layer was not re-initialised, so nothing reset at all'


def test_scope_all_wipes_the_wide_layer():
    net = QRDuelingNet(N_ACTIONS, n_quantiles=51, lin_size=64, stride=1)
    wide_before = net.value[0].weight.detach().clone()
    reset_last_linear(net, (net.value, net.adv), 'all')
    assert not torch.equal(wide_before, net.value[0].weight.detach())


def test_the_conv_stack_is_too_small_to_be_an_encoder():
    """The premise behind the fix, pinned so it is not taken on trust.

    If someone widens the trunk later this fails, which is the signal that
    'reset the head' can go back to meaning the streams.
    """
    net = get_estimator(N_ACTIONS, lin_size=64, stride=1)
    conv = sum(p.numel() for p in net[0].parameters())
    total = sum(p.numel() for p in net.parameters())
    assert conv < 5_000
    assert conv / total < 0.001

    # receptive field of kernels 3, 2, 2 at stride 1
    rf, jump = 1, 1
    for k, s in ((3, 1), (2, 1), (2, 1)):
        rf += (k - 1) * jump
        jump *= s
    assert rf == 5
    assert rf < 64 / 9, 'a conv feature cannot see a whole Crafter tile'


def test_reset_is_counted_correctly():
    """The printed percentage has to be the real one."""
    net = QRDuelingNet(N_ACTIONS, n_quantiles=51, lin_size=64, stride=1)
    touched, total = reset_last_linear(net, (net.value, net.adv), 'last')
    expected = sum(
        m.weight.numel() + m.bias.numel()
        for stream in (net.value, net.adv)
        for m in [[x for x in stream if isinstance(x, nn.Linear)][-1]])
    assert touched == expected
    assert total == sum(p.numel() for p in net.parameters())
