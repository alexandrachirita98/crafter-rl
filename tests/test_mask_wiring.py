"""The mask must reach the agent, and must change nothing when it is off.

test_masked_step_never_picks_illegal is the one that matters: it drives a real
agent over a real VecEnv and checks the actions against the mask it was given.
A mask that is computed and then dropped on the floor would pass every test in
test_masking.py.
"""

import numpy as np
import pytest
import torch

from common.env import N_ACTIONS, VecEnv
from common.masking import legal_masks
import stage_02_dqn.train as s02


def _agent(num_envs, warmup=20):
    net = s02.get_estimator(N_ACTIONS, lin_size=64, stride=2).to(s02.DEVICE)
    return s02.DQN(net, s02.ReplayMemory(size=2000, batch_size=8),
                   torch.optim.Adam(net.parameters(), lr=1e-4),
                   s02.get_epsilon_schedule(start=1.0, end=0.05, steps=100),
                   N_ACTIONS, gamma=0.99, warmup_steps=warmup, update_steps=2,
                   update_target_steps=50, num_envs=num_envs)


def test_masked_step_never_picks_illegal():
    """Drive the real agent and hold every action against its own mask."""
    s02.reset_rng(3)
    agent = _agent(4)
    env = VecEnv(num_envs=4, seed=5, length=60)
    states = env.reset()
    masks, checked = None, 0

    for _ in range(200):
        actions = agent.step(states, masks)
        if masks is not None:
            for i, a in enumerate(actions):
                assert masks[i][a] > 0, (
                    f'step() chose action {a}, which its mask forbade')
                checked += 1
        states, rewards, dones, infos = env.step(actions)
        for i in range(4):
            agent.learn(i, states[i], actions[i], rewards[i], states[i],
                        s02.bootstrap_mask(dones[i], infos[i]))
        masks = legal_masks(infos, dones)

    assert checked > 700, f'only checked {checked} actions'


def test_mask_actually_forbids_something():
    """A mask of all ones would pass the test above trivially."""
    env = VecEnv(num_envs=4, seed=5, length=60)
    env.reset()
    illegal = 0
    for _ in range(100):
        _, _, dones, infos = env.step(np.random.randint(N_ACTIONS, size=4))
        illegal += int((legal_masks(infos, dones) == 0.0).sum())
    assert illegal > 1000, (
        f'only {illegal} illegal actions over 400 env-steps; the mask is '
        'barely restricting anything and the experiment has no lever')


def test_apply_mask_is_identity_when_all_legal():
    """Why an unmasked run stays bit-identical: q + 0 * -1e8 is exactly q."""
    q = torch.tensor([[1.5, 1.5, 0.3], [0.1, -2.0, 0.1]])
    out = s02._apply_mask(q, np.ones((2, 3), dtype=np.float32))
    assert torch.equal(q, out)


def test_apply_mask_beats_the_largest_legal_value():
    q = torch.tensor([[1.0, 99.0, 2.0]])
    masked = s02._apply_mask(q, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    assert masked.argmax(1).item() == 2, 'the forbidden 99 still won'
    assert torch.isfinite(masked).all(), 'masking produced inf/nan'


def test_step_without_masks_matches_unmasked_rng():
    """masks=None must consume the RNG exactly as the old code did.

    The old step() drew np.random.randint(action_num, size=n). If random_legal
    on an all-ones mask drew a different number of values, every run recorded
    before masking would stop reproducing.
    """
    ones = np.ones((4, N_ACTIONS), dtype=np.float32)
    np.random.seed(0)
    old = np.random.randint(N_ACTIONS, size=4)
    after_old = np.random.rand()
    np.random.seed(0)
    new = s02.random_legal(ones)
    after_new = np.random.rand()
    assert list(old) == list(new)
    assert after_old == after_new, 'the RNG stream diverged'


@pytest.mark.parametrize('stage_mod', [
    'stage_02_dqn', 'stage_03_double', 'stage_04_nstep', 'stage_05_dueling',
    'stage_06_qrdqn', 'stage_07_munchausen', 'stage_08_per',
    'stage_09_ez_greedy', 'stage_10_drq_resets', 'stage_10c_no_per',
])
def test_every_stage_step_accepts_masks(stage_mod):
    """A stage whose step() ignores masks would silently train unmasked."""
    import importlib
    import inspect
    mod = importlib.import_module(f'{stage_mod}.train')
    # The agent class is the one main() is told to instantiate, so take the
    # DQN subclasses and skip VecEnv, whose step() is the environment's.
    agents = [obj for obj in vars(mod).values()
              if inspect.isclass(obj) and issubclass(obj, s02.DQN)]
    assert agents, f'no DQN subclass found in {stage_mod}'
    for cls in agents:
        params = inspect.signature(cls.step).parameters
        assert 'masks' in params, (
            f'{stage_mod}.{cls.__name__}.step() has no masks parameter; '
            'masking would be silently dropped')
