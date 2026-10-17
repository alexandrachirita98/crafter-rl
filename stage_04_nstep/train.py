"""Stage 04 -- n-step returns on top of stage 03's Double DQN.

Sutton 1988; Sutton & Barto 2018 ch. 7.

One-step learning moves reward information back one step per gradient update.
Crafter pays +1 once for an achievement and then nothing for hundreds of
steps, so a one-step agent has to crawl that signal back link by link. Summing
the next n real rewards before bootstrapping moves it n steps per update:

    G = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
        + gamma^n * Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))

Three things change: the buffer holds six-tuples (s, a, G, s_n, mask, n),
_store keeps a sliding window, and _update discounts by gamma ** n using the n
stored on each sample.

Memory is unchanged: the window holds s_0 and s_n, two frames per transition,
the same as stage 02's (s, s_).

Run:
    python stage_04_nstep/train.py --test-nstep
    python stage_04_nstep/train.py --total-steps 2000 --warmup-steps 500 \
        --buffer-size 5000
    python stage_04_nstep/train.py --total-steps 200000 --seed 1
"""

import pathlib
import random
import sys
from collections import deque

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (DEVICE, N_ACTIONS, ReplayMemory, _to_tensor,
                                get_estimator, main as stage_02_main)
from stage_03_double.train import DoubleDQN

STAGE_ID = 'stage_04_nstep'


class NStepReplayMemory(ReplayMemory):
    """The same deque, six-tuples: (s, a, G, s_n, mask, n)."""

    def sample(self):
        s, a, g, s_, m, n = zip(*random.sample(self._buffer, self._batch_size))
        col = lambda v, dt: torch.tensor(v, dtype=dt).unsqueeze(1).to(DEVICE)
        return (_to_tensor(np.stack(s)), col(a, torch.int64),
                col(g, torch.float32), _to_tensor(np.stack(s_)),
                col(m, torch.float32), col(n, torch.float32))


class NStepDoubleDQN(DoubleDQN):

    def __init__(self, *args, n_step=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_step = n_step
        self._window = deque()          # the last <= n steps, oldest first

    def _store(self, state, action, reward, state_, mask):
        self._window.append((state, action, reward, state_, mask))
        if mask == 0.0:                 # died: every open window ends here
            self._flush()
        elif len(self._window) == self._n_step:
            self._emit()

    def end_episode(self):
        self._flush()                   # the 10,000-step cap, mask still 1.0

    def _emit(self):
        """Pop the oldest step and push its n-step transition."""
        G = 0.0
        for k, (_, _, r, _, _) in enumerate(self._window):
            G += self._gamma ** k * r
        s0, a0 = self._window[0][0], self._window[0][1]
        _, _, _, s_n, mask_n = self._window[-1]
        self._buffer.push((s0, a0, G, s_n, mask_n, len(self._window)))
        self._window.popleft()

    def _flush(self):
        while self._window:
            self._emit()

    def _update(self, states, actions, rewards, states_, mask, n):
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            qsa_ = self._target_estimator(states_).gather(1, actions_)
        qsa = self._estimator(states).gather(1, actions)

        # n is (B,1) and differs per sample: a window cut short by death
        # accumulated fewer rewards, so it discounts by gamma ** its own n.
        # Using gamma ** self._n_step for every row is the classic n-step bug
        # and it does not crash.
        target_qsa = rewards + (self._gamma ** n) * qsa_ * mask

        loss = (qsa - target_qsa).pow(2).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


def extra_args(p):
    p.add_argument('--n-step', type=int, default=10,
                   help='rewards summed before bootstrapping. DER uses 20')
    p.add_argument('--test-nstep', action='store_true',
                   help='run the window self-test and exit')


def build(args, capacity):
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = NStepReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {'n_step': args.n_step}


# ------------------------------------------------------------- self-test


class _StubBuffer:
    """Catches pushes so the window logic can be checked without a net."""

    def __init__(self):
        self.pushed = []
        self._batch_size = 1

    def push(self, transition):
        self.pushed.append(transition)


def test_nstep(gamma=0.9, n_step=3):
    """Scripted episode: rewards 1 0 0 1 0, death on the last step.

    Checks both halves of the contract -- the flush on death, and the n stored
    per sample. Expected, with s_k named by index:

        from  G     s_n  mask  n
        s0    1.00  s3   1     3
        s1    0.81  s4   1     3
        s2    0.90  s5   0     3
        s3    1.00  s5   0     2
        s4    0.00  s5   0     1
    """
    agent = NStepDoubleDQN.__new__(NStepDoubleDQN)
    agent._gamma, agent._n_step = gamma, n_step
    agent._window = deque()
    agent._buffer = _StubBuffer()

    rewards = [1.0, 0.0, 0.0, 1.0, 0.0]
    masks = [1.0, 1.0, 1.0, 1.0, 0.0]          # died on step 5
    for t, (r, m) in enumerate(zip(rewards, masks)):
        agent._store(f's{t}', t, r, f's{t + 1}', m)

    expected = [('s0', 1.00, 's3', 1.0, 3), ('s1', 0.81, 's4', 1.0, 3),
                ('s2', 0.90, 's5', 0.0, 3), ('s3', 1.00, 's5', 0.0, 2),
                ('s4', 0.00, 's5', 0.0, 1)]
    print(f'death flush, gamma={gamma}, n={n_step}')
    print(f'{"from":>5} {"G":>6} {"s_n":>5} {"mask":>5} {"n":>3}')
    ok = True
    assert len(agent._buffer.pushed) == len(expected), agent._buffer.pushed
    for (s0, a, g, sn, m, n), (es0, eg, esn, em, en) in zip(
            agent._buffer.pushed, expected):
        good = (s0 == es0 and abs(g - eg) < 1e-9 and sn == esn
                and m == em and n == en)
        ok &= good
        print(f'{s0:>5} {g:6.2f} {sn:>5} {m:5.1f} {n:3d} '
              f'{"ok" if good else "WRONG, want " + str((es0, eg, esn, em, en))}')
    assert not agent._window, 'death must leave the window empty'

    # the 10,000-step cap: done with mask still 1.0, so only end_episode()
    # can cut the window. Drop this and the agent never trains on the last
    # n steps of a capped episode.
    agent._buffer.pushed.clear()
    for t in (10, 11):
        agent._store(f's{t}', t, 0.0, f's{t + 1}', 1.0)
    agent.end_episode()
    tail = [(p[4], p[5]) for p in agent._buffer.pushed]
    print(f'\ncap flush: (mask, n) = {tail}, want [(1.0, 2), (1.0, 1)]')
    ok &= tail == [(1.0, 2), (1.0, 1)]
    assert not agent._window

    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-nstep' in (argv if argv is not None else sys.argv[1:]):
        return test_nstep()
    return stage_02_main(argv, agent_cls=NStepDoubleDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
