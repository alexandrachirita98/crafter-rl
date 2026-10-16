"""Replay buffer: uint8 frames in host RAM.

Shared by every stage, so no stage's result can depend on how the buffer was
written. Later stages extend this file:
  stage 04 -- n-step returns
  stage 08 -- prioritised replay (sum-tree)
  stage 12 -- a second, never-overwritten buffer of demonstrations

THE MEMORY DECISION, which dictates where training can run at all.

  64*64*3 = 12,288 bytes per frame, so a 1M-frame buffer is 12.3 GB.
  Fits on RunPod (32 GB+). Does NOT fit locally (about 4 GB free), which is
  why local runs cap --buffer-size at 100k or less.

Two choices keep it to 12 GB rather than 49 GB:

  1. uint8, not float32. The /255 conversion happens on the GPU inside
     NatureCNN.forward(). float32 here would be 4x the RAM for zero benefit.
  2. ONE obs array, not obs AND next_obs. next_obs of step t IS obs of
     step t+1, so we store obs once and read index i+1 when we need the
     successor. Storing both would double 12 GB to 24 GB for nothing.
"""

import numpy as np

from common.env import OBS_SHAPE


def bytes_for(capacity, obs_shape=OBS_SHAPE):
    """Footprint of a buffer of this capacity, without allocating it.

    Worth calling before a run rather than after: on a box that cannot hold
    the buffer, allocating it to measure it is itself the crash.
    """
    per_frame = int(np.prod(obs_shape))          # uint8 obs
    per_frame += 8 + 4 + 4 + 1                   # action, reward, mask, done
    return capacity * per_frame


class ReplayBuffer:
    """Circular buffer of transitions, sampled uniformly at random.

    Why a buffer at all (Mnih et al. 2015): consecutive Crafter frames are
    nearly identical, so training on them in the order they arrive makes the
    network chase its own tail. Random batches of old transitions decorrelate
    the updates.
    """

    def __init__(self, capacity, obs_shape=OBS_SHAPE):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        # mask is info['discount'] -- 0.0 only on actual death, 1.0 on the
        # 10,000-step timeout. See common.env.bootstrap_mask.
        self.mask = np.zeros(self.capacity, dtype=np.float32)
        # done is the episode boundary, death OR timeout. Used only to refuse
        # to sample across it; never as the bootstrap mask.
        self.done = np.zeros(self.capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def nbytes(self):
        """Actual allocation of this buffer."""
        return sum(a.nbytes for a in
                   (self.obs, self.action, self.reward, self.mask, self.done))

    def add(self, obs, action, reward, mask, done):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.mask[i] = mask
        self.done[i] = done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _invalid(self, idx):
        """Indices whose successor frame is not the real next observation.

        TWO RULES, AND EACH IS A REAL BUG IF SKIPPED. Both produce an agent
        that ALMOST learns, which is the hardest kind of bug to find.

        1. Never sample an index with done=True. The frame at i+1 there is the
           first frame of a DIFFERENT episode in a freshly generated world --
           that teaches the agent that chopping a tree teleports it.

        2. Never sample the index whose successor has not been written. That
           is ptr-1 while filling, and once the buffer has wrapped it is also
           ptr-1, because obs[ptr] is then the OLDEST surviving frame, not the
           successor of ptr-1.
        """
        return self.done[idx] | (idx == (self.ptr - 1) % self.capacity)

    def sample(self, batch_size, rng):
        """Uniform batch of (obs, action, reward, next_obs, mask).

        next_obs is read at i+1, which is why the rules above exist.
        """
        if self.size < 2:
            raise ValueError('buffer needs at least 2 transitions to sample')
        idx = rng.integers(0, self.size, size=batch_size)
        bad = self._invalid(idx)
        # Rejection sampling. The rejected fraction is tiny -- one index per
        # episode plus one for the write head -- so this converges immediately.
        while bad.any():
            idx[bad] = rng.integers(0, self.size, size=int(bad.sum()))
            bad = self._invalid(idx)
        nxt = (idx + 1) % self.capacity
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.obs[nxt], self.mask[idx])


def test_buffer():
    """Check the two sampling rules hold, including after a wrap.

    Run: python -m common.replay
    """
    rng = np.random.default_rng(0)
    capacity = 64
    buf = ReplayBuffer(capacity, obs_shape=(4, 4, 3))

    # Fill past capacity so the buffer wraps, with an episode end every 7
    # steps. Frame i is filled with the value i % 256 so that a sampled
    # next_obs can be checked against the obs it should follow.
    n_added = 150
    for i in range(n_added):
        frame = np.full((4, 4, 3), i % 256, dtype=np.uint8)
        buf.add(frame, i % 17, float(i), 1.0, done=(i % 7 == 6))
    assert len(buf) == capacity
    assert buf.ptr == n_added % capacity

    obs, act, rew, nxt, mask = buf.sample(2000, rng)
    assert obs.shape == (2000, 4, 4, 3), obs.shape
    assert act.shape == rew.shape == mask.shape == (2000,)

    # Rule 1+2 together: the successor frame must be exactly one step later.
    # Frame values are consecutive mod 256, so this catches both a sample
    # across an episode boundary and a sample of the stale write head.
    first = obs[:, 0, 0, 0].astype(np.int64)
    second = nxt[:, 0, 0, 0].astype(np.int64)
    assert np.all((second - first) % 256 == 1), 'next_obs is not obs[i+1]'

    # And none of the sampled transitions may be an episode end.
    for o, r in zip(first, rew):
        assert int(r) % 7 != 6, 'sampled a done=True transition'

    # Small buffer, pre-wrap: the write head must still be excluded.
    small = ReplayBuffer(32, obs_shape=(4, 4, 3))
    for i in range(5):
        small.add(np.full((4, 4, 3), i, dtype=np.uint8), 0, float(i), 1.0, False)
    o, a, r, n, m = small.sample(500, rng)
    assert np.all(r < 4), 'sampled the newest index, whose next_obs is unwritten'

    # Report the 1M footprint WITHOUT allocating it -- 12 GB does not fit on
    # the local box, which is the whole reason this number matters.
    print(f'sampled 2000 transitions across a wrap, all successors valid')
    print(f'1M-frame buffer would allocate {bytes_for(1_000_000) / 1e9:.1f} GB')
    print('buffer OK')


if __name__ == '__main__':
    test_buffer()
