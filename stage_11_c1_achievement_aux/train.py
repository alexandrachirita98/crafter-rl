"""Stage 11 -- an auxiliary head predicting what unlocks next.

Our own addition, following UNREAL (Jaderberg et al. 2017,
https://arxiv.org/abs/1611.05397). Reward arrives maybe once every few hundred
steps, which is a starvation diet for a conv trunk that has to learn to see
trees, stone and zombies -- but the environment hands us the achievement state
every step, free, and nothing is training on it.

So: a second head on the shared trunk, 23-way classification of which
achievement unlocks within the next `horizon` steps (class 22 = none soon). To
predict "a stone pickaxe is coming" the trunk must represent "I hold wood and
I am near stone", which is the perception the main task needs. Action
selection is untouched; only the gradient reaching the trunk changes.

Labels come from looking forward in the replay buffer. PPO throws experience
away after a few epochs, which is why Moon et al. 2023 needed an episodic
memory for the same trick; the buffer lets us skip that.

Two implementation notes. The usual recipe indexes labels out of the buffer
after the fact, but ours is a ring of opaque tuples with no stable index into
the future, so labels are computed at store time from a lookahead window kept
alongside stage 04's n-step window. And the tuple grows to seven --
(s, a, G, s_n, mask, n, y) -- carried by both the n-step deque and the
prioritised ring.

Run:
    python stage_11_c1_achievement_aux/train.py --test-aux
    python stage_11_c1_achievement_aux/train.py --total-steps 200000 --seed 1 \
        --lambda-aux 0.1
"""

import pathlib
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import bootstrap_mask
from common.masking import legal_mask
from common.metrics import ACHIEVEMENTS
from common.replay import beta_by_step
from stage_02_dqn.train import (DEVICE, N_ACTIONS, _to_tensor, get_estimator,
                                main as stage_02_main)
from stage_04_nstep.train import NStepDoubleDQN, NStepReplayMemory
from stage_06_qrdqn.train import QRDuelingNet
from stage_08_per.train import PrioritizedReplayMemory
from stage_10_drq_resets.train import (DrQResets, build as build_10,
                                       extra_args as extra_10, random_shift)

STAGE_ID = 'stage_11_c1_achievement_aux'

N_ACH = len(ACHIEVEMENTS)       # 22
NONE_SOON = N_ACH               # class 22, the "nothing happens" bucket
N_AUX_CLASSES = N_ACH + 1       # 23


# ------------------------------------------------------- 1. the labeller


class AchievementLabeller:
    """Turns the per-step achievement counts into "what unlocks next" labels.

    Holds a lookahead window; a transition is emitted once we have seen
    `horizon` further steps or the episode ended. Never look across an episode
    boundary -- the next episode is a different world.

    Counts are cumulative within an episode, so "newly unlocked" is
    count[t + k] > count[t], not count[t + k] > 0.
    """

    def __init__(self, horizon=100):
        self._horizon = horizon
        self._pending = deque()     # (payload, counts_at_that_step)
        self._future = deque()      # counts for the steps already seen

    def push(self, payload, counts):
        """Queue one transition. Returns the list of (payload, label) ready."""
        self._pending.append((payload, counts))
        self._future.append(counts)
        out = []
        # The oldest pending item has `len(self._future) - 1` steps of future
        # behind it; emit it once that reaches the horizon.
        while len(self._future) > self._horizon:
            payload0, base = self._pending.popleft()
            self._future.popleft()
            out.append((payload0, self._label(base)))
        return out

    def flush(self):
        """Episode over. Emit everything on the shorter future we have."""
        out = []
        while self._pending:
            payload0, base = self._pending.popleft()
            self._future.popleft()
            out.append((payload0, self._label(base)))
        self._future.clear()
        return out

    def _label(self, base):
        """First achievement whose count rises above `base` within the window."""
        for counts in self._future:
            newly = counts > base
            if newly.any():
                return int(np.argmax(newly))
        return NONE_SOON


def counts_vector(info):
    """info['achievements'] -> the 22-vector, in common.metrics order."""
    ach = info['achievements']
    return np.array([ach[name] for name in ACHIEVEMENTS], dtype=np.int32)


# ----------------------------------------------------------- 2. the net


class AuxQRDuelingNet(QRDuelingNet):
    """Stage 06's net plus a 23-way head on the same trunk.

    forward() is untouched, so every stage 02-10 code path -- step(),
    _per_sample, _target_quantiles -- behaves exactly as before. The aux head
    is reached only through aux_logits().
    """

    def __init__(self, action_num, n_quantiles=51, lin_size=64, stride=1,
                 aux_hidden=256):
        super().__init__(action_num, n_quantiles=n_quantiles,
                         lin_size=lin_size, stride=stride)
        flat = self.value[0].in_features       # stage 02's probed width
        self.aux = nn.Sequential(
            nn.Linear(flat, aux_hidden), nn.ReLU(inplace=True),
            nn.Linear(aux_hidden, N_AUX_CLASSES))

    def aux_logits(self, x):
        return self.aux(self.trunk(x))


class AuxQNet(nn.Module):
    """Stage 02's net with a 23-way head on its penultimate features.

    AuxQRDuelingNet puts the head on the conv flatten -- 57,600 wide at
    stride 1, so that head alone is 14,751,767 params against 22,186,893 for
    the whole net. Here it sits after the Linear(flat, lin) + ReLU instead,
    so the shared representation is conv + first linear and the head is
    lin x 256 x 23: 22,551 params inside 3,712,648. forward() is exactly
    get_estimator's, so the RL path is stage 02's.
    """

    def __init__(self, action_num, lin_size=64, stride=1, aux_hidden=256):
        super().__init__()
        net = get_estimator(action_num, lin_size=lin_size, stride=stride)
        # Sequential slicing returns a new container over the SAME modules,
        # so trunk and head share parameters with net; nothing is copied.
        self.trunk = net[:-1]           # conv -> Linear(flat, lin) -> ReLU
        self.head = net[-1]             # Linear(lin, action_num)
        self.aux = nn.Sequential(
            nn.Linear(lin_size, aux_hidden), nn.ReLU(inplace=True),
            nn.Linear(aux_hidden, N_AUX_CLASSES)).to(DEVICE)

    def forward(self, x):
        return self.head(self.trunk(x))

    def aux_logits(self, x):
        return self.aux(self.trunk(x))


# -------------------------------------------------------- 3. the buffers


class AuxNStepMixin:
    """Seven-tuples: the stage-04 six plus the aux label."""

    def _unpack(self, rows):
        s, a, g, s_, m, n, y = zip(*rows)
        col = lambda v, dt: torch.tensor(
            np.asarray(v), dtype=dt).unsqueeze(1).to(DEVICE)
        return (_to_tensor(np.stack(s)), col(a, torch.int64),
                col(g, torch.float32), _to_tensor(np.stack(s_)),
                col(m, torch.float32), col(n, torch.float32),
                torch.tensor(np.asarray(y), dtype=torch.int64).to(DEVICE))


class AuxPrioritizedReplayMemory(AuxNStepMixin, PrioritizedReplayMemory):
    """Stage 08's ring and sum-tree, carrying the label through sample()."""

    def sample(self):
        B, total = self._batch_size, self._tree.total()
        prefix = (np.arange(B) + np.random.rand(B)) * total / B
        idx = self._tree.find(prefix)
        p = self._tree.nodes[self._tree.size + idx] / total
        w = (self._len * p) ** -self.beta
        w = w / w.max()
        cols = self._unpack([self._buffer[i] for i in idx])
        weights = torch.tensor(w, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        return (*cols, idx, weights)


class AuxNStepReplayMemory(AuxNStepMixin, NStepReplayMemory):
    """Stage 04's deque of six-tuples, carrying the label as a seventh field.

    Uniform sampling; no tree, no weights. _unpack already knows the seven
    columns.
    """

    def sample(self):
        return self._unpack(random.sample(self._buffer, self._batch_size))


# ---------------------------------------------------------- 4. the agent


class AuxMixin:
    """The labeller, and the routing of stage 04's emitted rows through it.

    Base-independent: everything here touches _windows, _buffer and
    _step_cnt, which every agent from stage 04 on has. The RL loss stays in
    the subclass, so the same labelling serves the quantile base and the
    scalar one.
    """

    def __init__(self, *args, lambda_aux=0.1, aux_horizon=100,
                 aux_none_weight=0.1, aux_wake_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lambda_aux = lambda_aux
        self._labeller = AchievementLabeller(horizon=aux_horizon)
        # Class imbalance, down-weighted rather than subsampled. Which class
        # dominates depends on the horizon, measured over 4k random steps:
        #
        #   horizon   none_soon   wake_up
        #        10       79.5%     12.2%
        #        25       52.2%     27.3%
        #        50       29.8%     53.0%
        #       100       22.7%     53.1%
        #
        # At the default horizon 100 the majority is not "none soon", it is
        # wake_up -- the player repeats it about twice an episode, so it ends
        # up being the next rise for most steps of the approach. Down-weighting
        # only class 22 there amplifies wake_up instead of damping it.
        weight = torch.ones(N_AUX_CLASSES, device=DEVICE)
        weight[NONE_SOON] = aux_none_weight
        if aux_wake_weight is not None:
            weight[ACHIEVEMENTS.index('wake_up')] = aux_wake_weight
        self._aux_weight = weight
        self._pending_counts = None     # set by learn() each step
        # logged alongside loss/meanQ
        self.aux_loss = float('nan')
        self.aux_acc = float('nan')
        self.aux_major = float('nan')

    # -- storing: route stage 04's emitted transitions through the labeller --

    def learn(self, i, state, action, reward, state_, mask, counts=None):
        """train_aux() passes the achievement counts for this step."""
        self._pending_counts = counts
        if self._step_cnt >= self._warmup_steps and len(self._buffer) == 0:
            # Stage 02 sizes warmup against the batch, but the n-step window
            # and the lookahead both hold transitions back, so a short run can
            # end warmup with nothing stored and sample unwritten rows.
            self._store(i, state, action, reward, state_, mask)
            self._step_cnt += 1
            return
        super().learn(i, state, action, reward, state_, mask)

    def _buffer_push(self, transition):
        """Hold stage 04's six-tuple until its label is known, then push."""
        counts = self._pending_counts
        if counts is None:                      # self-test path
            counts = np.zeros(N_ACH, dtype=np.int32)
        for payload, label in self._labeller.push(transition, counts):
            self._buffer.push((*payload, label))

    def _emit(self, i):
        """Stage 04's _emit, with the push redirected through the labeller."""
        window = self._windows[i]
        G = 0.0
        for k, (_, _, r, _, _) in enumerate(window):
            G += self._gamma ** k * r
        s0, a0 = window[0][0], window[0][1]
        _, _, _, s_n, mask_n = window[-1]
        self._buffer_push((s0, a0, G, s_n, mask_n, len(window)))
        window.popleft()

    def end_episode(self, i):
        super().end_episode(i)                   # flushes the n-step window
        # Labels for the tail are computed on a shorter future, which is
        # correct: the episode genuinely ends there.
        for payload, label in self._labeller.flush():
            self._buffer.push((*payload, label))


class AuxDrQResets(AuxMixin, DrQResets):
    """Stage 10, plus the auxiliary loss on the shared trunk."""

    def _update(self, states, actions, rewards, states_, mask, n, labels,
                idx, weights):
        self._buffer.beta = beta_by_step(self._step_cnt, self._total_steps)

        pred, per_sample = self._per_sample(states, actions, rewards,
                                            states_, mask, n)
        rl_loss = (weights.squeeze(1) * per_sample).mean()

        # The aux head sees the same augmented view the RL loss does, so the
        # trunk is not asked to be invariant for one task and not the other.
        logits = self._estimator.aux_logits(random_shift(states))
        aux_loss = F.cross_entropy(logits, labels, weight=self._aux_weight)

        (rl_loss + self._lambda_aux * aux_loss).backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._buffer.update_priorities(idx, per_sample.detach().cpu().numpy())
        self.loss = rl_loss.item()
        self.mean_q = pred.mean().item()
        self.aux_loss = aux_loss.item()
        with torch.no_grad():
            self.aux_acc = (logits.argmax(1) == labels).float().mean().item()
            # The baseline to beat is not 1/23, it is "always answer
            # whichever class is commonest in this batch" -- wake_up, not
            # none_soon, at the default horizon. Taking class 22's share here
            # would understate the bar.
            self.aux_major = (
                labels.bincount(minlength=N_AUX_CLASSES).max().item()
                / labels.numel())


class AuxNStepDoubleDQN(AuxMixin, NStepDoubleDQN):
    """C1 on the stage-03 base: scalar double-Q, n-step window, uniform replay.

    With --n-step 1 and --lambda-aux 0 the RL path is stage 03's _update to
    the line. One trunk pass feeds both heads, which is the point of sharing
    the representation.
    """

    def __init__(self, *args, total_steps=None, **kwargs):
        # Carried only so stage 13's --bbf-schedule can read it. The stage-10
        # base gets it from stage 08 (beta annealing); there is no PER here.
        super().__init__(*args, **kwargs)
        self._total_steps = total_steps

    def _update(self, states, actions, rewards, states_, mask, n, labels):
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            qsa_ = self._target_estimator(states_).gather(1, actions_)
        feats = self._estimator.trunk(states)
        qsa = self._estimator.head(feats).gather(1, actions)
        # n is (B,1) per sample; see stage 04 on why not gamma ** self._n_step
        target = rewards + (self._gamma ** n) * qsa_ * mask
        rl_loss = (qsa - target).pow(2).mean()

        logits = self._estimator.aux(feats)
        aux_loss = F.cross_entropy(logits, labels, weight=self._aux_weight)

        self._optimizer.zero_grad()
        (rl_loss + self._lambda_aux * aux_loss).backward()
        self._optimizer.step()

        self.loss = rl_loss.item()
        self.mean_q = qsa.mean().item()
        self.aux_loss = aux_loss.item()
        with torch.no_grad():
            self.aux_acc = (logits.argmax(1) == labels).float().mean().item()
            self.aux_major = (
                labels.bincount(minlength=N_AUX_CLASSES).max().item()
                / labels.numel())


# ----------------------------------------------------------- 5. the loop


def train_aux(agent, env, step_num=100_000, log_every=10_000,
              use_mask=False):
    """Stage 02's loop, passing info['achievements'] down to the labeller.

    use_mask turns on legal-action masking. This loop runs ONE env and calls
    reset() itself, so unlike the VecEnv loop there is no stale-info seam --
    but the mask must still be dropped on reset, because reset() returns no
    info and the fresh player owns nothing.
    """
    N = 0
    ep_rewards = [0.0]
    # main() hands these stages a VecEnv(num_envs=1), not a bare crafter.Env,
    # so everything here is a length-1 batch: reset() gives (1,64,64,3) and
    # step() wants a sequence of actions. Indexing [0] back out keeps the rest
    # of the loop scalar, which is what the labeller and the n-step window
    # expect.
    state, done = env.reset()[0], False
    start = time.time()
    mask = None                     # reset() returns no info

    for step in range(step_num):
        action = agent.step(state[None], mask)[0]
        state_, reward, done, info = env.step([action])
        state_, reward, done, info = state_[0], reward[0], done[0], info[0]
        if use_mask:
            mask = legal_mask(info)[None]
        agent.learn(0, state, action, reward, state_,
                    bootstrap_mask(done, info),
                    counts=counts_vector(info))
        state = state_
        ep_rewards[N] += reward

        if done:
            agent.end_episode(0)
            # VecEnv auto-resets, so state_ is already the new episode's
            # first frame; calling reset() here would throw it away and
            # start a third episode mid-stream.
            done = False
            mask = None             # new world, new player: nothing owned
            ep_rewards.append(0.0)
            N += 1

        if (step + 1) % log_every == 0:
            fps = (step + 1) / (time.time() - start)
            recent = ep_rewards[-11:-1] or [float('nan')]
            print('[{0:4d}][{1:8d}], R/ep={2:6.2f}, eps={3:4.2f}, '
                  'loss={4:8.4f}, meanQ={5:7.3f}, aux={6:6.3f}, '
                  'acc={7:4.1%} (majority {8:4.1%}), {9:5.0f} fps'.format(
                      N, step + 1, float(np.mean(recent)), agent.eps,
                      agent.loss, agent.mean_q, agent.aux_loss,
                      agent.aux_acc, agent.aux_major, fps), flush=True)

    return None, N, time.time() - start


def extra_args(p):
    extra_10(p)
    p.add_argument('--base', choices=tuple(BASES), default='stage03',
                   help='stage03: scalar double-DQN + uniform replay, the '
                        'agent that scores. stage10: the Rainbow stack. '
                        'The stage-10 flags are accepted and ignored on '
                        'stage03')
    p.add_argument('--lambda-aux', type=float, default=0.1,
                   help='weight on the auxiliary loss. 0 is the control')
    p.add_argument('--aux-horizon', type=int, default=100,
                   help='steps to look ahead for the next unlock')
    p.add_argument('--aux-none-weight', type=float, default=0.1,
                   help='cross-entropy weight on class 22 ("none soon")')
    p.add_argument('--aux-wake-weight', type=float, default=None,
                   help='cross-entropy weight on wake_up; it is the real '
                        'majority class at horizon >= 50. Default: unweighted')
    p.add_argument('--test-aux', action='store_true',
                   help='check the labeller and one aux gradient, then exit')


def build(args, capacity):
    _, _, extra = build_10(args, capacity)
    net = AuxQRDuelingNet(N_ACTIONS, n_quantiles=args.n_quantiles,
                          lin_size=args.lin_size, stride=args.stride).to(DEVICE)
    buffer = AuxPrioritizedReplayMemory(size=capacity,
                                        batch_size=args.batch_size,
                                        alpha=args.per_alpha)
    return net, buffer, dict(extra, lambda_aux=args.lambda_aux,
                             aux_horizon=args.aux_horizon,
                             aux_none_weight=args.aux_none_weight,
                             aux_wake_weight=args.aux_wake_weight)


def build_base03(args, capacity):
    """The stage-03 base: plain net, uniform n-step buffer, plus the head.

    Nothing from stages 05-10 is constructed here, so --n-quantiles,
    --per-alpha, --ez-mu and --reset-every have no effect on this base.
    """
    net = AuxQNet(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = AuxNStepReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, dict(n_step=args.n_step,
                             total_steps=args.total_steps,
                             lambda_aux=args.lambda_aux,
                             aux_horizon=args.aux_horizon,
                             aux_none_weight=args.aux_none_weight,
                             aux_wake_weight=args.aux_wake_weight)


BASES = {
    'stage03': (AuxNStepDoubleDQN, build_base03),
    'stage10': (AuxDrQResets, build),
}


def pick_base(argv):
    """main() needs the agent class before parse_args() has run."""
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument('--base', choices=tuple(BASES), default='stage03')
    return p.parse_known_args(argv if argv is not None
                              else sys.argv[1:])[0].base


# ------------------------------------------------------------- self-test


def test_aux(horizon=3):
    """Two halves: the labeller's arithmetic, then a real gradient step.

    The labeller is the easy part to get silently wrong. An off-by-one in the
    window, or a label leaking across an episode boundary, gives a head that
    trains happily on garbage.
    """
    ok = True

    # -- half one: scripted counts, hand-checked labels -------------------
    # 6 steps. collect_wood (index 6) rises at step 2, place_table (20) at
    # step 5. With horizon=3, step t sees the counts of steps t..t+3.
    names = {n: i for i, n in enumerate(ACHIEVEMENTS)}
    seq = []
    c = np.zeros(N_ACH, dtype=np.int32)
    for t in range(6):
        if t == 2:
            c = c.copy(); c[names['collect_wood']] += 1
        if t == 5:
            c = c.copy(); c[names['place_table']] += 1
        seq.append(c)

    lab = AchievementLabeller(horizon=horizon)
    got = []
    for t, c in enumerate(seq):
        got += [(p, y) for p, y in lab.push(f't{t}', c)]
    got += lab.flush()

    want = [('t0', names['collect_wood']),   # wood at t2, within 3
            ('t1', names['collect_wood']),   # wood at t2, within 3
            ('t2', names['place_table']),    # table at t5, exactly 3 ahead
            ('t3', names['place_table']),
            ('t4', names['place_table']),
            ('t5', NONE_SOON)]               # nothing after it
    print(f'labeller, horizon={horizon}, wood at t2 and table at t5')
    print(f'{"step":>5} {"label":>22} {"want":>22}')
    for (p, y), (wp, wy) in zip(got, want):
        good = p == wp and y == wy
        ok &= good
        nm = ACHIEVEMENTS[y] if y < N_ACH else 'none_soon'
        wnm = ACHIEVEMENTS[wy] if wy < N_ACH else 'none_soon'
        print(f'{p:>5} {nm:>22} {wnm:>22} {"ok" if good else "WRONG"}')
    ok &= len(got) == len(want)

    # Episode boundaries: flush must empty everything, so the next episode's
    # unlocks can never label this episode's states.
    empty = not lab._pending and not lab._future
    print(f'\nflush empties the window   {empty}   want True')
    ok &= empty

    # -- half two: the head trains and the trunk feels it ------------------
    import argparse
    import torch.optim as O
    from stage_02_dqn.train import get_epsilon_schedule, reset_rng

    reset_rng(0)
    steps, warmup, capacity, batch = 40, 20, 64, 8
    args = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                              n_step=3, n_quantiles=11, per_alpha=0.5,
                              total_steps=steps, ez_mu=2.0, ez_max_repeat=100,
                              reset_every=10_000, lambda_aux=0.5,
                              aux_horizon=3, aux_none_weight=0.1,
                              aux_wake_weight=None)
    net, buffer, extra = build(args, capacity)
    agent = AuxDrQResets(net, buffer, O.Adam(net.parameters(), lr=1e-4),
                         get_epsilon_schedule(1.0, 0.05, steps), N_ACTIONS,
                         gamma=0.99, warmup_steps=warmup, update_steps=2,
                         update_target_steps=10, **extra)

    rng = np.random.RandomState(0)
    counts = np.zeros(N_ACH, dtype=np.int32)
    trunk0 = net.trunk[1].weight.detach().clone()
    for t in range(steps):
        if t % 7 == 0:                       # something unlocks now and then
            counts = counts.copy(); counts[t % N_ACH] += 1
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.learn(0, s, t % N_ACTIONS, float(t % 2), s_, 1.0, counts=counts)

    shape_ok = net.aux_logits(
        torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8,
                      device=DEVICE)).shape == (batch, N_AUX_CLASSES)
    trained = not np.isnan(agent.aux_loss)
    trunk_moved = not torch.equal(trunk0, net.trunk[1].weight.detach())
    labels_vary = len({row[6] for row in buffer._buffer if row is not None}) > 1
    print(f'\naux_logits shape           {shape_ok}   want True  '
          f'(B, {N_AUX_CLASSES})')
    print(f'aux loss computed          {trained}   want True  '
          f'({agent.aux_loss:.3f})')
    print(f'gradient reached the trunk {trunk_moved}   want True')
    print(f'buffer holds varied labels {labels_vary}   want True')
    ok &= shape_ok and trained and trunk_moved and labels_vary

    # -- half three: the same, on the stage-03 base ------------------------
    # The scalar base shares no code with the quantile one below _emit, so
    # "the labeller works" does not imply "the head trains here too".
    reset_rng(0)
    args03 = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                                n_step=3, total_steps=steps, lambda_aux=0.5,
                                aux_horizon=3, aux_none_weight=0.1,
                                aux_wake_weight=None)
    net3, buf3, extra3 = build_base03(args03, capacity)
    agent3 = AuxNStepDoubleDQN(net3, buf3, O.Adam(net3.parameters(), lr=1e-4),
                               get_epsilon_schedule(1.0, 0.05, steps),
                               N_ACTIONS, gamma=0.99, warmup_steps=warmup,
                               update_steps=2, update_target_steps=10,
                               **extra3)
    rng = np.random.RandomState(0)
    counts = np.zeros(N_ACH, dtype=np.int32)
    trunk0 = net3.trunk[1].weight.detach().clone()
    for t in range(steps):
        if t % 7 == 0:
            counts = counts.copy(); counts[t % N_ACH] += 1
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent3.learn(0, s, t % N_ACTIONS, float(t % 2), s_, 1.0,
                     counts=counts)

    rl_ok = not np.isnan(agent3.loss)
    aux_ok = not np.isnan(agent3.aux_loss)
    moved3 = not torch.equal(trunk0, net3.trunk[1].weight.detach())
    width = len(buf3._buffer[0]) == 7
    # The whole reason for this base: stage 02's net, not stage 06's.
    n_par = sum(q.numel() for q in net3.parameters())
    print(f'\nstage-03 base')
    print(f'  rl loss computed         {rl_ok}   want True  '
          f'({agent3.loss:.3f})')
    print(f'  aux loss computed        {aux_ok}   want True  '
          f'({agent3.aux_loss:.3f})')
    print(f'  gradient reached trunk   {moved3}   want True')
    print(f'  buffer holds seven-tuples {width}   want True')
    print(f'  params {n_par:,} (scalar net, no quantiles/dueling)')
    ok &= rl_ok and aux_ok and moved3 and width

    # -- half four: lambda_aux = 0 is a true control -----------------------
    # The whole 200k gate rests on this: if any aux gradient leaks through at
    # lambda_aux=0, "stage 11 reproduces stage 03" is not a claim we can make.
    reset_rng(0)
    args0 = argparse.Namespace(lin_size=16, stride=2, batch_size=batch,
                               n_step=1, total_steps=steps, lambda_aux=0.0,
                               aux_horizon=3, aux_none_weight=0.1,
                               aux_wake_weight=None)
    net0, buf0, extra0 = build_base03(args0, capacity)
    agent0 = AuxNStepDoubleDQN(net0, buf0, O.Adam(net0.parameters(), lr=1e-3),
                               get_epsilon_schedule(1.0, 0.05, steps),
                               N_ACTIONS, gamma=0.99, warmup_steps=warmup,
                               update_steps=2, update_target_steps=10,
                               **extra0)
    aux_w0 = net0.aux[0].weight.detach().clone()
    head_w0 = net0.head.weight.detach().clone()
    rng = np.random.RandomState(0)
    counts = np.zeros(N_ACH, dtype=np.int32)
    for t in range(steps):
        if t % 7 == 0:
            counts = counts.copy(); counts[t % N_ACH] += 1
        s = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        s_ = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent0.learn(0, s, t % N_ACTIONS, float(t % 2), s_, 1.0,
                     counts=counts)
    aux_frozen = torch.equal(aux_w0, net0.aux[0].weight.detach())
    q_trains = not torch.equal(head_w0, net0.head.weight.detach())
    print(f'\nlambda_aux=0 is a true control')
    print(f'  aux head never moves     {aux_frozen}   want True')
    print(f'  Q head still trains      {q_trains}   want True')
    print(f'  aux loss still reported  {not np.isnan(agent0.aux_loss)}   '
          f'want True ({agent0.aux_loss:.3f}, logged, zero gradient)')
    ok &= aux_frozen and q_trains
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-aux' in (argv if argv is not None else sys.argv[1:]):
        return test_aux()
    if '--num-envs' in (argv if argv is not None else sys.argv[1:]):
        # These stages replace stage 02's loop with a single-env one, so a
        # parallel request must fail loudly rather than be ignored.
        import argparse as _ap
        _p = _ap.ArgumentParser(add_help=False)
        _p.add_argument('--num-envs', type=int, default=1)
        if _p.parse_known_args(argv if argv is not None else sys.argv[1:])[0].num_envs > 1:
            raise SystemExit(
                f'{STAGE_ID} runs a single-env loop (its labeller needs one '
                'continuous trajectory); --num-envs > 1 is not supported')

    import stage_02_dqn.train as s02
    original = s02.train
    s02.train = train_aux               # the loop must pass achievements down
    agent_cls, build_fn = BASES[pick_base(argv)]
    try:
        return stage_02_main(argv, agent_cls=agent_cls, stage_id=STAGE_ID,
                             extra_args=extra_args, build=build_fn)
    finally:
        s02.train = original


if __name__ == '__main__':
    sys.exit(main() or 0)
