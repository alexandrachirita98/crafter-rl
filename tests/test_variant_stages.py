"""Tests for the stage 14-17 variants from docs/beyond-c1-c2-c3.md.

Each stage also carries a --test-* self-test that run.sh invokes. These cover
the failure modes that train happily and learn the wrong thing: stage 16's
bonus paying per run instead of per episode or not decaying to 0, and stage
17 pairing inv_ with s instead of s_.
"""

import importlib

import numpy as np
import pytest
import torch

s14 = importlib.import_module('stage_14_extended_mask.train')
s15 = importlib.import_module('stage_15_encoder.train')
s16 = importlib.import_module('stage_16_reward_shaping.train')
s17 = importlib.import_module('stage_17_inventory_input.train')

from common.env import N_ACTIONS


# ------------------------------------------------- the stages' own self-tests


def test_stage14_selftest():
    assert s14.test_extmask()


def test_stage15_selftest():
    assert s15.test_encoder()


def test_stage16_selftest():
    assert s16.test_shaping()


def test_stage17_selftest():
    assert s17.test_invinput()


# --------------------------------------------------------- stage 15 encoders


@pytest.mark.parametrize('kind', s15.ENCODERS)
def test_encoder_takes_nchw_uint8(kind):
    """Every arm must accept stage 02's tensor convention and return (B,17)."""

    class _Args:
        lin_size = 64
        stride = 1

    net = s15.make_encoder(kind, _Args())
    obs = torch.randint(0, 256, (3, 3, 64, 64), dtype=torch.uint8).to(s15.DEVICE)
    with torch.no_grad():
        out = net(obs)
    assert out.shape == (3, N_ACTIONS)
    assert torch.isfinite(out).all()


def test_encoder_param_counts_are_what_the_doc_claims():
    """The argument for this stage rests on these numbers, so pin them."""

    class _Args:
        lin_size = 64
        stride = 1

    counts = {k: sum(p.numel() for p in s15.make_encoder(k, _Args()).parameters())
              for k in s15.ENCODERS}
    assert counts['lab'] == 3_690_097
    assert counts['stride2'] == 413_169
    assert counts['nature'] == 609_457
    # The whole claim: NatureCNN is smaller than the champion.
    assert counts['nature'] < counts['lab'] / 5


# ---------------------------------------------------------- stage 16 shaping


def _inv(**kw):
    base = {k: 0 for k in ('stone', 'coal', 'iron', 'diamond', 'wood')}
    base.update(kw)
    return {'inventory': base}


def test_shaping_pays_once_per_episode_not_once_per_run():
    a = s16._stub()
    a.shape(0, 0.0, _inv(), 0)
    assert a.shape(0, 0.0, _inv(stone=1), 0) == pytest.approx(0.1)
    assert a.shape(0, 0.0, _inv(stone=2), 0) == 0.0
    s16.ShapedDoubleDQN.end_episode(a, 0)
    a.shape(0, 0.0, _inv(), 0)
    assert a.shape(0, 0.0, _inv(stone=1), 0) == pytest.approx(0.1), \
        'a new episode must restore the entitlement'


def test_shaping_cannot_be_farmed_by_spending():
    """Crafting spends stone. Re-collecting it must not re-pay."""
    a = s16._stub()
    a.shape(0, 0.0, _inv(), 0)
    a.shape(0, 0.0, _inv(stone=1), 0)
    a.shape(0, 0.0, _inv(stone=0), 0)       # spent on a pickaxe
    assert a.shape(0, 0.0, _inv(stone=1), 0) == 0.0


def test_shaping_decays_to_exactly_zero():
    a = s16._stub(decay=300_000, total=1_000_000)
    assert a._bonus_now(0) == pytest.approx(0.1)
    assert a._bonus_now(700_000) == pytest.approx(0.1)
    assert a._bonus_now(1_000_000) == 0.0, \
        'the final policy must optimise the TRUE reward'


def test_shaping_bonus_zero_is_a_true_control():
    a = s16._stub(bonus=0.0)
    a.shape(0, 0.0, _inv(), 0)
    assert a.shape(0, 2.0, _inv(stone=1, coal=1, iron=1), 0) == 2.0
    assert a.shaped_total == 0.0


def test_shaping_envs_are_independent():
    a = s16._stub(num_envs=2)
    a.shape(0, 0.0, _inv(), 0)
    a.shape(1, 0.0, _inv(), 0)
    a.shape(0, 0.0, _inv(stone=1), 0)
    assert a.shape(1, 0.0, _inv(stone=1), 0) == pytest.approx(0.1), \
        "env 0's stone must not silence env 1's"


# ------------------------------------------------- stage 17 inventory input


def test_inv_vector_order_is_frozen():
    """The order is baked into the net's input layer."""
    assert s17.INV_KEYS[0] == 'health'
    assert s17.N_INV == 16
    info = {'inventory': {k: 0 for k in s17.INV_KEYS}}
    info['inventory']['iron_sword'] = 9
    v = s17.inv_vector(info)
    assert v[s17.INV_KEYS.index('iron_sword')] == pytest.approx(1.0)
    assert v.sum() == pytest.approx(1.0)


def test_inv_vector_is_normalised_and_permissive():
    info = {'inventory': {k: 9 for k in s17.INV_KEYS}}
    v = s17.inv_vector(info)
    assert v.max() <= 1.0 and v.min() >= 0.0
    assert not s17.inv_vector(None).any()
    assert not s17.inv_vector({}).any()


def test_buffer_pairs_inv_with_s_and_inv_next_with_s_next():
    """THE ALIGNMENT TRAP. Mispairing trains the target on the wrong
    state's inventory and raises nothing."""
    buf = s17.InvReplayMemory(size=32, batch_size=4)
    for t in range(8):
        buf.push((
            np.full((64, 64, 3), t, dtype=np.uint8), 0, 0.0,
            np.full((64, 64, 3), t + 1, dtype=np.uint8), 1.0,
            np.ones(N_ACTIONS, dtype=np.float32),
            np.full(s17.N_INV, t / 100.0, dtype=np.float32),
            np.full(s17.N_INV, (t + 1) / 100.0, dtype=np.float32),
        ))
    states, _, _, states_, _, _, iv, iv_ = buf.sample()
    for r in range(4):
        t = int(states[r, 0, 0, 0].item())
        assert int(states_[r, 0, 0, 0].item()) == t + 1
        assert iv[r, 0].item() == pytest.approx(t / 100.0, abs=1e-5)
        assert iv_[r, 0].item() == pytest.approx((t + 1) / 100.0, abs=1e-5)


def test_side_input_changes_the_output_when_on_and_not_when_off():
    obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8).to(s17.DEVICE)
    zeros = torch.zeros(2, s17.N_INV, device=s17.DEVICE)
    ones = torch.ones(2, s17.N_INV, device=s17.DEVICE)

    on = s17.InvQNet(lin_size=64, use_inv=True).to(s17.DEVICE)
    with torch.no_grad():
        assert not torch.allclose(on(obs, zeros), on(obs, ones), atol=1e-6)

    off = s17.InvQNet(lin_size=64, use_inv=False).to(s17.DEVICE)
    with torch.no_grad():
        assert torch.allclose(off(obs, zeros), off(obs, ones), atol=1e-6)


def test_control_arm_has_the_same_parameter_count():
    """So --inv-input on/off differs in one input, not in net capacity."""
    on = sum(p.numel() for p in s17.InvQNet(lin_size=64, use_inv=True).parameters())
    off = sum(p.numel() for p in s17.InvQNet(lin_size=64, use_inv=False).parameters())
    assert on == off
