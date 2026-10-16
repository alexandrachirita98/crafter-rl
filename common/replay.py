"""Replay buffer: uint8 frames in host RAM.

Shared by every stage, so no stage's result can depend on how the buffer was
written. Two classes live here:

  SumTree                  -- O(log n) proportional sampling. Stage 08.
  ReplayBuffer             -- one-step transitions. Stages 01-03.
  NStepReplayBuffer        -- n-step returns, a subclass. Stage 04 onwards.
  PrioritizedReplayBuffer  -- ReplayBuffer + SumTree. Stage 08 onwards.

Later stages extend this file further:
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


def bytes_for(capacity, obs_shape=OBS_SHAPE, nstep=False, per=False):
    """Footprint of a buffer of this capacity, without allocating it.

    Worth calling before a run rather than after: on a box that cannot hold
    the buffer, allocating it to measure it is itself the crash.

    nstep=True prices NStepReplayBuffer, which stores a second frame array
    (next_obs) and a per-transition n. That is roughly 2x, and it is the
    number that decides whether a run fits in RAM from stage 04 onwards.

    per=True adds the stage-08 sum tree: 2*capacity float64 nodes, i.e. 16
    bytes per frame, and up to 2x that when capacity is just above a power of
    two, because SumTree rounds its leaf row up. Negligible either way against
    the 12,288 bytes of image -- worth pricing only so that nobody assumes
    prioritisation is what broke RAM.
    """
    per_frame = int(np.prod(obs_shape))          # uint8 obs
    per_frame += 8 + 4 + 4 + 1                   # action, reward, mask, done
    if nstep:
        per_frame += int(np.prod(obs_shape))     # next_obs, stored explicitly
        per_frame += 8                           # n, int64
    if per:
        per_frame += 16                          # sum tree, 2 float64 nodes
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


class NStepReplayBuffer(ReplayBuffer):
    """ReplayBuffer, but each slot holds an n-step return instead of one reward.

    Stage 04 onwards. See docs/implementation-guide.md section 4, stage 04.

    WHAT CHANGES AND WHY. The one-step buffer stores r_t and the learner
    bootstraps from s_{t+1}. Here the caller accumulates the next n real
    rewards first and stores

        G_t = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}

    alongside the observation n steps later. Reward information then travels n
    steps back per gradient update instead of one, which is the whole point on
    Crafter: an achievement is +1 once and then nothing for hundreds of steps.

    WHY `n` IS STORED PER TRANSITION, AND NOT AS A CONSTANT. This is the
    classic n-step bug. If the episode ended 4 steps into a 10-step window,
    that transition accumulated 4 rewards, so its bootstrap discount is
    gamma^4, not gamma^10. Storing one global n and using gamma^n everywhere
    silently mis-discounts every transition near a death -- and deaths are
    exactly where Crafter's reward signal lives. The learner must read

        target = G + (gamma ** n) * mask * next_q

    with the PER-TRANSITION n that sample() returns.

    WHY next_obs IS STORED, NOT READ AT i+1. The one-step buffer saves 12 GB
    by exploiting next_obs(t) == obs(t+1). That identity does not hold here:
    the successor of an n-step transition is the frame n steps later, which
    for a window cut short by death is not at a fixed offset and may have been
    overwritten by the time we sample it. So this buffer keeps its own
    next_obs array and costs 2x the RAM: 24.6 GB at 1M frames.

    THAT DOUBLING IS A REAL CONSTRAINT, not a footnote. Use bytes_for() with
    nstep=True before launching, and cap --buffer-size accordingly.
    """

    def __init__(self, capacity, obs_shape=OBS_SHAPE):
        super().__init__(capacity, obs_shape)
        # The two additions over the parent. next_obs is the frame n steps
        # later; n is how many real rewards went into this slot's return.
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.n = np.zeros(self.capacity, dtype=np.int64)

    def nbytes(self):
        return super().nbytes() + self.next_obs.nbytes + self.n.nbytes

    def add(self, *args, **kwargs):
        """Refused on purpose: a one-step add here would be a silent bug.

        The parent's add() writes no next_obs and no n, so the slot would
        train against a zero frame with gamma^0 = 1. Nothing would crash and
        the loss would still go down. Call add_nstep().
        """
        raise NotImplementedError(
            'NStepReplayBuffer stores n-step returns; call add_nstep()')

    def add_nstep(self, obs, action, ret, next_obs, mask, n, done):
        """Store one n-step transition.

        Args:
          obs:      s_t, the frame the window STARTS at.
          action:   a_t, the action taken at s_t. Not the later ones.
          ret:      G_t, the discounted sum of the n rewards in the window.
          next_obs: s_{t+n}, the frame the window ENDS at.
          mask:     bootstrap mask of the LAST step in the window -- 0.0 only
                    if the player died inside it. From common.env.
          n:        how many real rewards are in `ret`. 1 <= n <= n_step.
          done:     True if the window ends an episode. Only used to keep
                    sample() from treating the slot as mid-episode.
        """
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action
        self.reward[i] = ret
        self.mask[i] = mask
        self.n[i] = n
        self.done[i] = done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Uniform batch of (obs, action, G, next_obs, mask, n).

        No rejection sampling, and no _invalid(): both of the parent's rules
        existed only because next_obs was read at i+1. Every slot here carries
        its own successor frame, so every written slot is valid -- including
        the write head and including episode ends.
        """
        if self.size < 1:
            raise ValueError('buffer is empty')
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.mask[idx], self.n[idx])


class SumTree:
    """Fixed-size array heap where each parent holds the SUM of its children.

    Stage 08. Schaul et al. 2016, "Prioritized Experience Replay",
    https://arxiv.org/abs/1511.05952, Appendix B.2.1.

    WHY A TREE AND NOT np.random.choice(p=priorities). The naive version is
    one line and it is the thing that kills the run. Proportional sampling
    over a 1M-element probability vector costs O(n) per batch: numpy has to
    normalise and scan the whole array. At 1M frames and one gradient step per
    env step that is a million-element pass 1M times -- hours of pure
    bookkeeping on top of the training. The tree answers both operations the
    algorithm needs in O(log n):

        total()            -- the sum of all priorities, the root
        find(prefix_sum)    -- which leaf does this cumulative mass land in?
        update(i, p)       -- change one leaf and fix its ancestors

    At capacity 1M that is 20 comparisons per sample instead of 1M.

    THE LAYOUT. One float64 array of 2*size entries, no Python objects:

        nodes[0]                unused, keeps the index arithmetic clean
        nodes[1]                root = sum of every priority
        nodes[size .. 2*size-1] the leaves, leaf i is nodes[size + i]
        children of k           nodes[2k], nodes[2k+1]
        parent of k             nodes[k // 2]

    `size` IS THE CAPACITY ROUNDED UP TO A POWER OF TWO, and that is not
    cosmetic. find() descends a fixed number of levels because the tree is
    COMPLETE -- every path from root to leaf is the same length, so one
    vectorised loop with no per-element bookkeeping does the whole batch. With
    a capacity like 2000 the leaf row does not start at a power of two, the
    fixed-depth descent runs one level too far, and indexing walks off the end
    of the array. (It did: a 2000-step smoke test with --buffer-size 5000
    crashed with `index 4002 out of bounds` before this was rounded.) The
    padding leaves sit at priority 0.0 forever and are therefore unreachable,
    which test_sum_tree() checks directly.

    float64 IS DELIBERATE. Priorities are summed a million at a time and then
    repeatedly decremented during descent; in float32 the accumulated rounding
    drifts far enough that find() can walk off the end of the populated
    leaves and return an index that was never written. 16 MB at 1M capacity
    is a rounding error next to the 12 GB of frames.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        # Round up to a power of two so the tree is complete; see THE LAYOUT.
        # `size` is the leaf-row offset used by every index calculation here,
        # and it is what callers index the leaf row by.
        self.size = 1
        while self.size < self.capacity:
            self.size *= 2
        self.nodes = np.zeros(2 * self.size, dtype=np.float64)

    def total(self):
        """Sum of every leaf priority. The root, read in O(1)."""
        return float(self.nodes[1])

    def max_leaf(self):
        """Largest single priority currently stored, or 0.0 if empty.

        Used by the buffer to give a brand-new transition the maximum
        priority seen so far, so that every transition is trained on at
        least once before priority has any say. See the buffer's add().
        """
        return float(self.nodes[self.size:].max())

    def update(self, idx, priority):
        """Set leaf `idx` to `priority` and repair every ancestor sum.

        idx and priority may be arrays; duplicate indices in one call are
        handled by looping, because np.add.at-style scatter would double-count
        a repeated leaf. Duplicates are normal -- a batch can sample the same
        transition twice.
        """
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        priority = np.atleast_1d(np.asarray(priority, dtype=np.float64))
        if np.any(priority < 0):
            raise ValueError('priorities must be non-negative')
        for i, p in zip(idx, priority):
            k = self.size + int(i)
            delta = p - self.nodes[k]
            self.nodes[k] = p
            # Walk to the root adding the delta. Adding the delta rather than
            # recomputing nodes[k] = nodes[2k] + nodes[2k+1] is the same
            # answer in exact arithmetic and half the memory traffic.
            k //= 2
            while k >= 1:
                self.nodes[k] += delta
                k //= 2

    def find(self, prefix):
        """Leaf index whose cumulative priority range contains `prefix`.

        Descend from the root: if the left child holds at least as much mass
        as we are still looking for, go left; otherwise subtract the left
        child's mass and go right. Vectorised over a batch of prefixes, so
        one batch of 64 is 64 parallel descents of 20 steps, not 64 loops.

        `prefix` must lie in [0, total()). A prefix at or above total()
        descends to the last leaf rather than raising, which is the behaviour
        we want on the float64 edge case where a stratified draw lands exactly
        on the total.
        """
        prefix = np.atleast_1d(np.asarray(prefix, dtype=np.float64)).copy()
        k = np.ones(prefix.shape, dtype=np.int64)
        # Fixed depth: the tree is complete, so every descent is exactly
        # log2(capacity) levels and no per-element while loop is needed.
        while k[0] < self.size:
            left = 2 * k
            left_mass = self.nodes[left]
            # >= , NOT > . Each leaf owns the HALF-OPEN range [cum, cum+p), so
            # a prefix landing exactly on a boundary belongs to the leaf on the
            # right. With a strict `>` a zero-priority leaf owns the single
            # point at its own boundary, and prefix=0.0 against a buffer whose
            # first leaf is still 0.0 descends straight into it -- returning a
            # transition that was never written. Verified in test_sum_tree().
            go_right = prefix >= left_mass
            prefix = np.where(go_right, prefix - left_mass, prefix)
            k = left + go_right.astype(np.int64)
        return k - self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """ReplayBuffer, but a transition's sampling chance grows with its TD error.

    Stage 08. Schaul et al. 2016, https://arxiv.org/abs/1511.05952 --
    proportional variant (their eq. 1), the one Rainbow uses.

    THE IDEA IN ONE LINE. Uniform sampling spends most of its gradient steps
    on frames the network already predicts correctly. Sample instead with

        P(i) = p_i^alpha / sum_k p_k^alpha,    p_i = |TD error of i| + eps

    so a surprising transition -- the one frame where the tree actually
    became wood -- gets re-studied until it stops being surprising.

    AND THE BIAS IT INTRODUCES, WHICH IS NOT OPTIONAL TO FIX. The Bellman
    loss is an expectation over the buffer's own distribution. Changing which
    transitions arrive changes what the loss is an expectation OF, so the
    network converges to the wrong fixed point: high-error transitions are
    over-weighted twice, once by arriving often and once by having a big
    gradient. Importance sampling weights undo exactly the first:

        w_i = (1 / (N * P(i)))^beta,  normalised by max(w) in the batch

    and are applied to the PER-SAMPLE loss, never to the mean. beta anneals
    0.4 -> 1.0 over training (paper section 3.4): early on the bias is less
    harmful than the noise the full correction adds, and by the end the
    correction is complete.

    WHY NORMALISE BY max(w). Without it the weights scale the whole loss by a
    factor that changes with the priority distribution, which silently
    rescales the effective learning rate as training proceeds. Dividing by
    the batch maximum puts every weight in (0, 1] and makes the largest
    update the reference. (Paper section 3.4.)

    WHAT THIS COSTS, MEASURED. Two extra float64 arrays of 2*capacity: 16 MB
    per million frames against the 12.3 GB of images, so RAM is unchanged for
    practical purposes. The real cost is wall-clock -- see bytes_for() and the
    timing in test_per() below.

    THE THREE WAYS THIS GOES WRONG AND STILL RUNS:

    1. New transitions added with priority 0 are never sampled, so the newest
       experience is invisible until something else touches it. add() gives
       them max priority instead.
    2. update_priorities() called with the WRONG indices -- e.g. the batch's
       position rather than the buffer index sample() returned -- trains
       fine and prioritises random transitions. sample() returns the indices
       for exactly this reason; pass them straight back.
    3. The IS weights folded into F.smooth_l1_loss's own reduction. The
       weight is per-sample, so the loss must be computed with
       reduction='none' and then weighted and meaned by hand.

    EXPECT NOTHING FROM THIS STAGE. docs/contribution.md section 4.2: EDE,
    the only DQN-family agent that does well on Crafter, DELETED prioritised
    replay and got 2.7x better. We build it because Contribution 3 (stage 12)
    needs the machinery -- demo transitions get a permanent priority bonus,
    Hester et al. 2018 section 3 -- and a null result here is the EXPECTED
    outcome, not a bug to hunt.
    """

    def __init__(self, capacity, obs_shape=OBS_SHAPE, alpha=0.5, eps=1e-6):
        super().__init__(capacity, obs_shape)
        # alpha = 0 is uniform sampling, alpha = 1 is fully proportional.
        # 0.5 is the proportional-variant value from the paper's section 3.3
        # and Rainbow's table 1. Stored on the buffer, not passed per call,
        # so a run cannot change it halfway.
        self.alpha = float(alpha)
        # eps keeps a transition with a perfectly-predicted TD error of 0.0
        # reachable. Without it such a transition is permanently unsamplable
        # -- including, early on, almost every transition.
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)
        # The priority a brand-new transition gets. Tracked rather than read
        # from the tree because max_leaf() is O(capacity); this is O(1) and
        # only ever grows, which is the conservative direction.
        self._max_priority = 1.0

    def nbytes(self):
        return super().nbytes() + self.tree.nodes.nbytes

    def add(self, obs, action, reward, mask, done):
        """Store a transition at maximum priority.

        MAXIMUM, not its true TD error, because we do not have the TD error
        yet -- computing it would need a forward pass per env step. The paper
        (section 3.3) does the same: new transitions go in at the current
        maximum so that every transition is guaranteed at least one visit,
        after which its real error takes over. Priority 0 here would make the
        newest experience invisible.
        """
        i = self.ptr                         # read BEFORE the parent moves it
        super().add(obs, action, reward, mask, done)
        self.tree.update(i, self._max_priority ** self.alpha)

    def sample(self, batch_size, rng, beta=0.4):
        """Prioritised batch plus the IS weights and buffer indices.

        Returns (obs, action, reward, next_obs, mask, weights, idx).
        `idx` must be handed back to update_priorities() after the TD errors
        are known -- that is the whole loop.

        STRATIFIED, NOT PLAIN, SAMPLING. Split [0, total) into batch_size
        equal segments and draw one uniform inside each (paper's appendix
        B.2.1). Plain uniform draws over the whole range would let one
        very-high-priority transition fill most of the batch, which both
        wastes the batch and spikes the gradient. Stratifying guarantees the
        batch spans the whole priority range.
        """
        if self.size < 2:
            raise ValueError('buffer needs at least 2 transitions to sample')
        total = self.tree.total()
        if total <= 0.0:
            raise ValueError('every priority is zero; nothing to sample')

        segment = total / batch_size
        offsets = (np.arange(batch_size) + rng.random(batch_size)) * segment
        idx = self.tree.find(offsets)

        # The parent's two sampling rules still hold -- next_obs is still read
        # at i+1 here -- but rejection sampling by redrawing would break the
        # stratification. Resample the offending segments instead, which
        # keeps each draw inside its own stratum.
        bad = self._invalid(idx)
        for _ in range(100):
            if not bad.any():
                break
            n_bad = int(bad.sum())
            redraw = (np.arange(batch_size)[bad] + rng.random(n_bad)) * segment
            idx[bad] = self.tree.find(redraw)
            bad = self._invalid(idx)
        if bad.any():
            # A stratum that is entirely invalid cannot be fixed by redrawing
            # inside it -- with a tiny buffer, or one stratum sitting exactly
            # on an episode end, the redraw returns the same leaf every time.
            # Walk BACKWARDS to the nearest valid slot rather than return a
            # transition whose next_obs is the first frame of another world.
            #
            # Note the modulus is self.size, which is also the number of
            # written slots: before a wrap the live range is [0, size) and
            # after a wrap size == capacity, so one expression covers both.
            # Stepping back once WITHOUT re-checking (the obvious one-liner)
            # lands on another done=True slot whenever two episode ends are
            # adjacent, which is why this loops.
            for _ in range(self.size):
                if not bad.any():
                    break
                idx[bad] = (idx[bad] - 1) % self.size
                bad = self._invalid(idx)
            if bad.any():
                raise ValueError(
                    'no valid transition in the buffer; every slot is an '
                    'episode end or the unwritten write head')

        # P(i) from the tree, then the IS weight. Note the exponent is -beta:
        # a transition sampled OFTEN (large P) gets a SMALL weight.
        probs = self.tree.nodes[self.tree.size + idx] / total
        weights = (self.size * np.maximum(probs, 1e-12)) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        nxt = (idx + 1) % self.capacity
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.obs[nxt], self.mask[idx], weights, idx)

    def update_priorities(self, idx, td_errors):
        """Write |TD error|^alpha back for the transitions just trained on.

        Call this EVERY time sample() is called, with the idx sample()
        returned. Skip it and the tree still holds the max-priority placeholder
        from add(), so sampling stays uniform-ish and the stage measures
        nothing -- a null result that looks exactly like the real null result
        this stage expects, which is why it would never be noticed.
        """
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self._max_priority = max(self._max_priority, float(priorities.max()))
        self.tree.update(idx, priorities ** self.alpha)


def beta_by_step(step, total_steps, beta_start=0.4, beta_end=1.0):
    """Linear anneal of the IS exponent, 0.4 -> 1.0 over the run.

    Schaul et al. 2016 section 3.4. The schedule is in terms of the FULL run
    length, so a 2000-step smoke test anneals over 2000 steps and a 1M run
    over 1M -- the shape of the curve is what the paper specifies, not an
    absolute step count.
    """
    frac = min(1.0, step / max(1, total_steps))
    return beta_start + frac * (beta_end - beta_start)


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


def test_nstep_buffer(gamma=0.99, n_step=10):
    """Check the n-step return and the PER-TRANSITION discount.

    The second of those is the classic n-step bug and the reason this test
    exists: a buffer that stores a global n passes every shape check and
    still mis-discounts every window that a death cut short.

    Run: python -m common.replay
    """
    rng = np.random.default_rng(0)

    # The window accumulator, exactly as stage_04_nstep/train.py uses it. Keep
    # the two in step: if the loop there changes, change it here too.
    from collections import deque

    def rollout(rewards, done_at):
        """Feed a synthetic episode through the accumulator, return the slots.

        rewards[t] is the reward for step t. done_at is the index of the last
        step of the episode, which is a DEATH here (mask 0.0).
        """
        buf = NStepReplayBuffer(256, obs_shape=(4, 4, 3))
        queue = deque(maxlen=n_step)
        for t, r in enumerate(rewards):
            frame = np.full((4, 4, 3), t % 256, dtype=np.uint8)
            done = (t == done_at)
            mask = 0.0 if done else 1.0
            nxt = np.full((4, 4, 3), (t + 1) % 256, dtype=np.uint8)
            queue.append((frame, t % 17, r, mask, nxt, done))
            if len(queue) == n_step:
                _push(buf, queue, gamma)
            if done:
                # Flush the tail: every partial window still in the queue is a
                # real transition and must reach the buffer, or the agent never
                # sees the n-1 steps that led to its own death. Push from the
                # CURRENT front first -- that window is only in the buffer
                # already if the queue had reached full length.
                if len(queue) < n_step:
                    _push(buf, queue, gamma)
                while len(queue) > 1:
                    queue.popleft()
                    _push(buf, queue, gamma)
                queue.clear()
        return buf

    def _push(buf, queue, g):
        G, mask_n, k, nxt = 0.0, 1.0, 0, None
        for k, (_, _, r, m, nx, _) in enumerate(queue):
            G += (g ** k) * r
            mask_n, nxt = m, nx
            if m == 0.0:
                break
        obs0, act0 = queue[0][0], queue[0][1]
        buf.add_nstep(obs0, act0, G, nxt, mask_n, n=k + 1,
                      done=(mask_n == 0.0))

    # --- case 1: a full window, no death ---------------------------------
    # 40 steps of reward 1.0, death at the very end. The first slot's window
    # is entirely interior, so G must be the full 10-term geometric sum.
    rewards = [1.0] * 40
    buf = rollout(rewards, done_at=39)
    expect_full = sum(gamma ** k for k in range(n_step))
    assert abs(float(buf.reward[0]) - expect_full) < 1e-5, \
        f'full-window return {buf.reward[0]} != {expect_full}'
    assert buf.n[0] == n_step, f'full window stored n={buf.n[0]}'
    assert buf.mask[0] == 1.0, 'interior window must still bootstrap'

    # obs of slot 0 is step 0; its successor must be step n_step, NOT step 1.
    assert int(buf.obs[0][0, 0, 0]) == 0
    assert int(buf.next_obs[0][0, 0, 0]) == n_step % 256, \
        'next_obs must be the frame n steps later'

    # --- case 2: THE BUG. a window cut short by death --------------------
    # Death at step 3 of a 10-step window. The slot starting at step 0 saw
    # only 4 rewards (steps 0,1,2,3), so n must be 4 and the mask 0.0.
    rewards = [1.0] * 12
    buf = rollout(rewards, done_at=3)
    assert buf.n[0] == 4, f'truncated window stored n={buf.n[0]}, expected 4'
    assert buf.mask[0] == 0.0, 'died inside the window: must not bootstrap'
    expect_trunc = sum(gamma ** k for k in range(4))
    assert abs(float(buf.reward[0]) - expect_trunc) < 1e-5, \
        f'truncated return {buf.reward[0]} != {expect_trunc}'
    # And the tail was flushed: steps 1,2,3 each got their own slot, with n
    # shrinking to 1 at the step of the death itself.
    assert len(buf) == 4, f'tail not flushed, {len(buf)} slots for 4 steps'
    assert list(buf.n[:4]) == [4, 3, 2, 1], f'n sequence {list(buf.n[:4])}'
    assert np.all(buf.mask[:4] == 0.0), 'every slot in a dying window: mask 0'

    # --- case 3: rewards must be DISCOUNTED, not summed ------------------
    # A reward only at the END of the window. Summing gives 1.0; discounting
    # gives gamma^(n-1). Catches a missing gamma**k.
    rewards = [0.0] * 9 + [1.0] + [0.0] * 30
    buf = rollout(rewards, done_at=39)
    assert abs(float(buf.reward[0]) - gamma ** 9) < 1e-6, \
        f'slot 0 return {buf.reward[0]} != gamma^9 = {gamma ** 9:.4f}'

    # --- case 4: sample() shapes and the n it returns --------------------
    rewards = [1.0] * 200
    buf = rollout(rewards, done_at=199)
    obs, act, ret, nxt, mask, n = buf.sample(500, rng)
    assert obs.shape == nxt.shape == (500, 4, 4, 3)
    assert act.shape == ret.shape == mask.shape == n.shape == (500,)
    assert np.all((n >= 1) & (n <= n_step)), 'n out of range'
    # Every sampled successor is exactly n steps after its obs.
    first = obs[:, 0, 0, 0].astype(np.int64)
    second = nxt[:, 0, 0, 0].astype(np.int64)
    assert np.all((second - first) % 256 == n), 'next_obs is not obs[i+n]'

    # --- case 5: the one-step add() must be refused ----------------------
    try:
        buf.add(np.zeros((4, 4, 3), np.uint8), 0, 0.0, 1.0, False)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('one-step add() on an n-step buffer must raise')

    print(f'n-step returns OK at n={n_step}: full windows, truncated windows, '
          f'tail flush, per-transition n')
    print(f'1M-frame n-step buffer would allocate '
          f'{bytes_for(1_000_000, nstep=True) / 1e9:.1f} GB '
          f'(vs {bytes_for(1_000_000) / 1e9:.1f} GB one-step)')
    print('n-step buffer OK')


def test_sum_tree():
    """Check find() really inverts the cumulative priority distribution.

    Run: python -m common.replay
    """
    tree = SumTree(8)
    priorities = [3.0, 1.0, 0.0, 2.0, 4.0, 0.0, 1.0, 1.0]   # total 12
    for i, p in enumerate(priorities):
        tree.update(i, p)

    assert abs(tree.total() - 12.0) < 1e-12, tree.total()
    assert abs(tree.max_leaf() - 4.0) < 1e-12

    # Hand-checked boundaries of the cumulative ranges:
    #   leaf 0: [0, 3)   leaf 1: [3, 4)   leaf 3: [4, 6)
    #   leaf 4: [6, 10)  leaf 6: [10, 11) leaf 7: [11, 12)
    # Leaves 2 and 5 have zero priority and must NEVER be returned.
    cases = [(0.0, 0), (2.9, 0), (3.0, 1), (3.5, 1), (4.0, 3), (5.9, 3),
             (6.0, 4), (9.9, 4), (10.0, 6), (11.0, 7), (11.9, 7)]
    for prefix, expected in cases:
        got = int(tree.find(prefix)[0])
        assert got == expected, f'find({prefix}) = {got}, expected {expected}'

    # Zero-priority leaves are unreachable: sample the whole range densely.
    dense = tree.find(np.linspace(0, 12, 10_001, endpoint=False))
    assert not np.isin(dense, [2, 5]).any(), 'zero-priority leaf was sampled'

    # The empirical frequency must match priority / total. 4.0/12 = 1/3 for
    # leaf 4; this is the property the whole stage rests on.
    counts = np.bincount(dense, minlength=8) / len(dense)
    for i, p in enumerate(priorities):
        assert abs(counts[i] - p / 12.0) < 1e-3, (i, counts[i], p / 12.0)

    # An update must repair the ancestors, not just the leaf.
    tree.update(2, 12.0)
    assert abs(tree.total() - 24.0) < 1e-12, tree.total()
    assert int(tree.find(13.0)[0]) == 2

    # NON-POWER-OF-TWO CAPACITY. The regression test for the crash the first
    # stage-08 smoke test hit: capacity 2000 made the fixed-depth descent run
    # one level past the leaf row. Every capacity must work, and the padding
    # leaves must stay unreachable.
    for capacity in (1, 3, 5, 2000, 5000):
        odd = SumTree(capacity)
        assert odd.size >= capacity and odd.size & (odd.size - 1) == 0
        for i in range(capacity):
            odd.update(i, float(i + 1))
        expected_total = capacity * (capacity + 1) / 2
        assert abs(odd.total() - expected_total) < 1e-9, capacity
        drawn = odd.find(np.linspace(0, expected_total, 2000, endpoint=False))
        assert drawn.min() >= 0 and drawn.max() < capacity, \
            f'capacity {capacity}: find() returned a padding leaf'

    print('sum tree: find() inverts the CDF, zero leaves unreachable,')
    print('          non-power-of-two capacities stay in bounds')


def test_per():
    """Check the four things that break prioritised replay silently."""
    rng = np.random.default_rng(0)
    capacity = 64
    buf = PrioritizedReplayBuffer(capacity, obs_shape=(4, 4, 3), alpha=0.5)

    n_added = 150
    for i in range(n_added):
        frame = np.full((4, 4, 3), i % 256, dtype=np.uint8)
        buf.add(frame, i % 17, float(i), 1.0, done=(i % 7 == 6))
    assert len(buf) == capacity

    obs, act, rew, nxt, mask, w, idx = buf.sample(256, rng, beta=0.4)
    assert obs.shape == (256, 4, 4, 3), obs.shape
    assert w.shape == idx.shape == (256,)

    # 1. The parent's two rules survive prioritisation: the successor frame is
    #    still exactly one step later, and no episode end is sampled.
    first = obs[:, 0, 0, 0].astype(np.int64)
    second = nxt[:, 0, 0, 0].astype(np.int64)
    assert np.all((second - first) % 256 == 1), 'next_obs is not obs[i+1]'
    assert np.all(np.asarray(rew, dtype=np.int64) % 7 != 6), 'sampled a done'

    # 2. IS weights are in (0, 1] with the max normalised to exactly 1.
    assert w.min() > 0.0 and w.max() <= 1.0 + 1e-6, (w.min(), w.max())
    assert abs(w.max() - 1.0) < 1e-6, 'weights not normalised by their max'

    # 3. Prioritisation actually bites, AND it bites in proportion. Flatten
    #    every priority, then give ONE valid transition a huge TD error, and
    #    check its share of the next batch matches its share of the tree's
    #    mass. Checking the PROPORTION rather than just "it appears a lot" is
    #    what catches a wrong alpha; checking it appears at all is what
    #    catches update_priorities() being wired to the batch position
    #    instead of the buffer index sample() returned.
    buf.update_priorities(np.arange(buf.size), np.full(buf.size, 1e-6))
    target = int(idx[0])
    buf.update_priorities([target], [100.0])
    leaves = buf.tree.nodes[buf.tree.size:]
    expected_share = leaves[target] / buf.tree.total()
    _, _, _, _, _, _, idx2 = buf.sample(4096, rng, beta=0.4)
    share = float((idx2 == target).mean())
    assert share > 0.9, f'high-priority transition only got {share:.1%} of batch'
    assert abs(share - expected_share) < 0.02, \
        f'share {share:.3f} does not match tree mass {expected_share:.3f}'

    # 4. beta anneals 0.4 -> 1.0 over the FULL run, and high-probability
    #    samples get smaller weights than low-probability ones.
    assert abs(beta_by_step(0, 1000) - 0.4) < 1e-9
    assert abs(beta_by_step(500, 1000) - 0.7) < 1e-9
    assert abs(beta_by_step(1000, 1000) - 1.0) < 1e-9
    assert abs(beta_by_step(5000, 1000) - 1.0) < 1e-9, 'beta must clamp at 1.0'

    _, _, _, _, _, w2, idx3 = buf.sample(256, rng, beta=1.0)
    hot = idx3 == target
    if hot.any() and (~hot).any():
        assert w2[hot].max() < w2[~hot].min(), \
            'the oversampled transition must get the SMALLER IS weight'

    # A fresh transition must be reachable immediately: priority 0 on add()
    # would make the newest experience invisible.
    buf.add(np.full((4, 4, 3), 200, dtype=np.uint8), 0, 999.0, 1.0, False)
    newest = (buf.ptr - 1) % buf.capacity
    assert buf.tree.nodes[buf.tree.size + newest] > 0.0, \
        'new transition entered the tree at priority 0'

    # 5. The invalid-index fallback. A buffer where almost every slot is an
    #    episode end is the only way to reach it, and it must still never
    #    return a done=True slot -- that is the bug that teaches the agent
    #    chopping a tree teleports it to a new world.
    dense = PrioritizedReplayBuffer(16, obs_shape=(4, 4, 3), alpha=0.5)
    for i in range(40):
        # done on every step except every 5th: adjacent episode ends, which
        # is what defeats a fallback that steps back only once.
        dense.add(np.full((4, 4, 3), i % 256, dtype=np.uint8), 0, float(i),
                  1.0, done=(i % 5 != 0))
    o5, _, _, n5, _, _, i5 = dense.sample(128, rng, beta=1.0)
    assert not dense.done[i5].any(), 'fallback returned an episode end'
    f5 = o5[:, 0, 0, 0].astype(np.int64)
    s5 = n5[:, 0, 0, 0].astype(np.int64)
    assert np.all((s5 - f5) % 256 == 1), 'fallback broke the successor frame'

    print(f'per: one hot transition took {share:.1%} of the batch for {expected_share:.1%}\n     of the tree mass; IS weights normalised to max 1.0')
    print(f'1M-frame PER buffer would allocate '
          f'{bytes_for(1_000_000, per=True) / 1e9:.1f} GB '
          f'(+{16e6 / 1e6:.0f} MB of tree over uniform)')
    print('per OK')


if __name__ == '__main__':
    test_buffer()
    test_sum_tree()
    test_per()
    print()
    test_nstep_buffer()
