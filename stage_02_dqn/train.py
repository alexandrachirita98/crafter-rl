"""Stage 02 -- plain DQN, from the course lab notebook 02_rainbow_sol.ipynb.

The lab's code, adapted where Crafter needs it:

  - Old Gym 4-tuple API: obs, reward, done, info; reset() returns obs alone.
  - done is True both on death and on the 10,000-step cap, and a timeout must
    still bootstrap, so transitions store info['discount'] as `mask` and the
    target multiplies by that instead of (1 - done).
  - The flatten width is measured, not hard-coded -- the lab's 9*16 is sized
    for MiniGrid's tile view.
  - ReplayMemory stores uint8 numpy frames; the lab's per-step GPU tensors
    don't scale to 1M.

MSE loss, no gradient clipping and the biased max target are the lab's and are
left alone -- stage 03 replaces the target, later stages the loss.

Run:
    python stage_02_dqn/train.py --total-steps 200000 --seed 1
"""

import argparse
import itertools
import pathlib
import random
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as O

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, OBS_SHAPE, VecEnv, bootstrap_mask
from common.masking import legal_masks, random_legal
from common.metrics import ACHIEVEMENTS, score_pooled

STAGE_ID = 'stage_02_dqn'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset_rng(seed=42):
    print(f'Setting all rngs to seed={seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------- 1. replay


class ReplayMemory:
    """The lab's deque buffer, holding uint8 numpy frames."""

    def __init__(self, size=1000, batch_size=32):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size

    def push(self, transition):
        self._buffer.append(transition)

    def sample(self):
        """-> states, actions, rewards, states_, mask. Frames are uint8."""
        s, a, r, s_, m = zip(*random.sample(self._buffer, self._batch_size))
        return (
            _to_tensor(np.stack(s)),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            _to_tensor(np.stack(s_)),
            torch.tensor(m, dtype=torch.float32).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self._buffer)


def _to_tensor(obs):
    """uint8 (64,64,3) frame or (N,64,64,3) batch -> (N,3,64,64) on device."""
    x = torch.from_numpy(np.ascontiguousarray(obs))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    # permute: Crafter gives NHWC, torch convs want NCHW. Getting this wrong
    # does not crash, it just learns nothing.
    # contiguous: permute only restrides, and View calls .view(), which needs
    # contiguous input.
    return x.permute(0, 3, 1, 2).contiguous().to(DEVICE)


def _mask_or_ones(masks, n, action_num):
    """All-ones when masking is off, so an unmasked run is bit-identical."""
    if masks is None:
        return np.ones((n, action_num), dtype=np.float32)
    return masks


def _apply_mask(qvals, masks):
    """-1e8 on illegal actions so argmax can never pick one.

    Additive rather than assignment because the stages differ in what they
    hand over: a plain (N,17) of Q-values, or the quantile mean. Both are
    finite, so -1e8 dominates without producing NaN the way -inf * 0 would.
    """
    mt = torch.as_tensor(masks, device=qvals.device, dtype=qvals.dtype)
    return qvals + (1.0 - mt) * -1e8


# -------------------------------------------------------------- 2. epsilon


def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    """Linear decay to `end` over `steps`, then flat. Consumed with next()."""
    eps_step = (start - end) / steps

    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step

    return itertools.chain(frange(start, end, eps_step),
                           itertools.repeat(end))


# -------------------------------------------------------------- 3. network


class ByteToFloat(nn.Module):
    def forward(self, x):
        assert x.dtype == torch.uint8, \
            'The model expects states of type ByteTensor.'
        return x.float().div_(255)


class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_estimator(action_num, input_ch=3, lin_size=32, stride=1):
    """The lab's 3-conv net, flatten width measured by a probe.

    stride=1 is the lab's architecture. On 64x64 input its convs barely
    downsample (-> 60x60), so the flatten is 57,600 and the first linear layer
    is 3.7M params. stride=2 gives 784 and ~54k, and runs ~5x faster. Default
    stays at the lab's value; report it in the ablation table if you change it.
    """
    conv = nn.Sequential(
        ByteToFloat(),
        nn.Conv2d(input_ch, 16, kernel_size=3, stride=stride),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=2, stride=stride),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=2, stride=stride),
        nn.ReLU(inplace=True),
        View(),
    )
    with torch.no_grad():
        probe = torch.zeros(1, input_ch, OBS_SHAPE[0], OBS_SHAPE[1],
                            dtype=torch.uint8)
        flat = conv(probe).shape[1]
    return nn.Sequential(
        conv,
        nn.Linear(flat, lin_size),
        nn.ReLU(inplace=True),
        nn.Linear(lin_size, action_num),
    ).to(DEVICE)


# ------------------------------------------------------------------ 4. DQN


class DQN:
    """The lab's DQN. learn() takes `mask` where the lab takes `done`."""

    def __init__(
        self,
        estimator,
        buffer,
        optimizer,
        epsilon_schedule,
        action_num,
        gamma=0.92,
        update_steps=4,
        update_target_steps=10,
        warmup_steps=100,
        num_envs=1,
    ):
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps = update_steps
        self._update_target_steps = update_target_steps
        self._warmup_steps = warmup_steps
        self._num_envs = num_envs
        self._step_cnt = 0
        # for the log line; NaN until the first gradient step
        self.loss = float('nan')
        self.mean_q = float('nan')
        self.eps = 1.0
        assert warmup_steps > self._buffer._batch_size, (
            'You should have at least a batch in the ER.')

    def step(self, states, masks=None):
        """Epsilon-greedy over a batch. (N,64,64,3) uint8 -> (N,) actions.

        masks is (N,17) with 1.0 on legal actions; None means no masking and
        reproduces every run recorded before masking existed. The mask applies
        to the random branch as well as the greedy one -- exploration is most
        of training, so leaving it unmasked would keep spending most presses on
        buttons that cannot fire.

        One forward pass for all N envs, not N passes: the batching is where
        the throughput comes from. The coin flip is per env -- a single shared
        flip would make every env explore or exploit in lockstep, which throws
        away the decorrelation that running them in parallel buys.

        The schedule advances once per call, not once per env, so epsilon
        tracks calls to step(). main() divides --eps-decay-steps by num_envs
        to keep it on the env-step clock.
        """
        n = len(states)
        masks = _mask_or_ones(masks, n, self._action_num)
        if self._step_cnt < self._warmup_steps:
            return random_legal(masks)

        self.eps = next(self._epsilon)
        greedy = np.random.rand(n) >= self.eps
        actions = random_legal(masks)
        if greedy.any():
            with torch.no_grad():
                qvals = self._estimator(_to_tensor(states[greedy]))
                qvals = _apply_mask(qvals, masks[greedy])
                actions[greedy] = qvals.argmax(1).cpu().numpy()
        return actions

    def learn(self, i, state, action, reward, state_, mask):
        """`i` is which env the transition came from. Stages that keep
        per-trajectory state (04's n-step window, 09's repeat walk) index by
        it; the replay buffer itself does not care."""
        self._store(i, state, action, reward, state_, mask)

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            batch = self._buffer.sample()
            self._update(*batch)

        # update the target estimator
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(
                self._estimator.state_dict())

        self._step_cnt += 1

    def _store(self, i, state, action, reward, state_, mask):
        """One-step: push as-is. Stage 04 keeps an n-step window here."""
        del i
        self._buffer.push((state, action, reward, state_, mask))

    def end_episode(self, i):
        """train() calls this when env `i` finishes. One-step does nothing."""
        del i

    def _update(self, states, actions, rewards, states_, mask):
        """Stage 03 overrides this method and nothing else."""
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        # mask is info['discount']: 0.0 only on death, so a timeout still
        # bootstraps. The lab's (1 - done) would zero it.
        target_qsa = rewards + self._gamma * qsa_ * mask

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


# ----------------------------------------------------------------- 5. loop


def train(agent, env, step_num=100_000, log_every=10_000, use_mask=False):
    """The lab's loop, over a VecEnv on Crafter's 4-tuple API.

    step_num is ENV STEPS, the benchmark budget, not loop iterations: with
    num_envs envs each iteration consumes num_envs of them, so the range
    strides by num_envs. Writing range(step_num) here would quietly run
    num_envs times the budget and make the number incomparable to every run
    already in */results/.

    VecEnv auto-resets, so a done env's next observation arrives in the same
    step() return. Do not call env.reset() on done -- the mask from
    bootstrap_mask is what stops the learner bootstrapping across the seam.

    Logs at a fixed interval rather than per episode, since Crafter episodes
    run to thousands of steps and stats.jsonl is the real record.

    use_mask turns on legal-action masking. It reads inventory and the
    semantic map out of info, which is privileged state -- not visible in the
    64x64 image the agent sees -- so a masked score is not strictly comparable
    to the pixels-only numbers in the Crafter paper. Off by default, which
    keeps every run recorded before masking existed reproducible.
    """
    num_envs = env.num_envs
    ep_reward = np.zeros(num_envs, dtype=np.float64)
    recent = deque(maxlen=10)
    states = env.reset()
    episodes = 0
    start = time.time()
    # reset() returns no info, so the first step runs unmasked -- which is
    # what legal_masks would give for a fresh player anyway.
    masks = None

    for step in range(0, step_num, num_envs):

        actions = agent.step(states, masks)

        states_, rewards, dones, infos = env.step(actions)

        if use_mask:
            # dones matters. VecEnv auto-resets and hands back the DYING
            # step's info beside the NEXT episode's first observation, so
            # masking the new state with it would forbid legal actions for a
            # whole episode's first step. legal_masks falls back to all-ones
            # wherever done is set.
            masks = legal_masks(infos, dones)

        for i in range(num_envs):
            agent.learn(i, states[i], actions[i], rewards[i], states_[i],
                        bootstrap_mask(dones[i], infos[i]))
            ep_reward[i] += rewards[i]
            if dones[i]:
                agent.end_episode(i)
                recent.append(ep_reward[i])
                ep_reward[i] = 0.0
                episodes += 1

        states = states_

        done_steps = step + num_envs
        if done_steps % log_every < num_envs:
            fps = done_steps / (time.time() - start)
            seen = list(recent) or [float('nan')]
            print('[{0:4d}][{1:8d}], R/ep={2:6.2f}, eps={3:4.2f}, '
                  'loss={4:8.4f}, meanQ={5:7.3f}, {6:5.0f} fps'.format(
                      episodes, done_steps,
                      float(np.mean(seen)), agent.eps,
                      agent.loss, agent.mean_q, fps), flush=True)

    return None, episodes, time.time() - start


# ------------------------------------------------------------------ 6. main


def parse_args(argv=None, extra_args=None):
    p = argparse.ArgumentParser(
        description='Stage 02: the course lab DQN, on Crafter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num-envs', type=int, default=1,
                   help='parallel Crafter envs sharing one replay buffer and '
                        'one net. 1 reproduces stages 02-10 exactly as run. '
                        '--total-steps stays the total across all envs')
    p.add_argument('--logdir', type=str, default='logdir')

    p.add_argument('--buffer-size', type=int, default=100_000,
                   help='transitions; 2 uint8 frames each, 100k = 2.5 GB')
    p.add_argument('--batch-size', type=int, default=32,
                   help="the lab's value")
    p.add_argument('--warmup-steps', type=int, default=20_000,
                   help='uniform-random actions, filling the buffer')
    p.add_argument('--update-steps', type=int, default=2,
                   help="gradient step every N env steps; the lab's value")
    p.add_argument('--update-target-steps', type=int, default=2000,
                   help='target-net refresh; DER value. The lab uses 8-256, '
                        'far too fast for a 1M budget')

    p.add_argument('--gamma', type=float, default=0.99,
                   help='the lab uses 0.92 on 100-step MiniGrid episodes')
    p.add_argument('--lr', type=float, default=1e-4,
                   help="the lab's 1e-3 diverges on a 64x64 input")
    p.add_argument('--adam-eps', type=float, default=1e-4,
                   help="the lab's value")
    p.add_argument('--lin-size', type=int, default=64,
                   help="width of the estimator's hidden layer")
    p.add_argument('--stride', type=int, default=1,
                   help="conv stride. 1 is the lab's net (3.7M params); "
                        '2 gives ~54k and runs ~5x faster')

    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-steps', type=int, default=250_000,
                   help='linear over the first 25%% of a 1M run')

    p.add_argument('--log-every', type=int, default=10_000)

    # Masking defaults OFF so every run already in */results/ reproduces.
    # --no-mask is the explicit control arm: it is the default, but naming it
    # on the command line puts it in command_seed<N>.txt, which is what the
    # summary reads back.
    mask = p.add_mutually_exclusive_group()
    mask.add_argument('--mask', dest='mask', action='store_true',
                      help='legal-action masking (privileged state; see '
                           'common/masking.py)')
    mask.add_argument('--no-mask', dest='mask', action='store_false',
                      help='disable legal-action masking (control arm)')
    p.set_defaults(mask=False)

    # A later stage adds its own flags here; see stage_04_nstep/train.py.
    if extra_args is not None:
        extra_args(p)
    return p.parse_args(argv)


def report(logdir, episodes):
    """Score the run. With num_envs > 1 the Recorders write one stats.jsonl
    per shard, so pool every episode and score the pool ONCE.

    Averaging per-shard scores is a different (and wrong) number: the score is
    a geometric mean over achievement rates, and each shard holds only
    1/num_envs of the episodes, where one lucky unlock reads as a far higher
    rate than it should. Pool first, score after.
    """
    shards = sorted(logdir.glob('shard*/stats.jsonl'))
    paths = shards or [logdir / 'stats.jsonl']
    paths = [q for q in paths if q.exists()]
    if not (paths and episodes):
        return
    score, percents, n = score_pooled(paths)
    print(f'score over {n} episodes: {score:.2f}%   (random baseline: ~1.6%)')
    print('\nper-achievement success rate, non-zero only:')
    for name, pct in sorted(zip(ACHIEVEMENTS, percents),
                            key=lambda kv: -kv[1]):
        if pct > 0:
            print(f'  {name:<20} {pct:5.1f}%')


def build(args, capacity):
    """Network, buffer and extra agent kwargs. Stage 02's choices.

    A later stage passes its own build to main() to swap the net or the
    buffer, and returns extra constructor kwargs in the third slot.
    """
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, {}


def main(argv=None, agent_cls=DQN, stage_id=STAGE_ID, extra_args=None,
         build=build):
    args = parse_args(argv, extra_args)

    logdir = (pathlib.Path(args.logdir) / f'crafter_reward-{stage_id}'
              / str(args.seed))
    logdir.mkdir(parents=True, exist_ok=True)

    reset_rng(args.seed)
    env = VecEnv(args.num_envs, logdir=logdir, seed=args.seed)

    capacity = min(args.buffer_size, args.total_steps)
    frame_bytes = int(np.prod(OBS_SHAPE))
    print(f'stage {stage_id}  seed {args.seed}  '
          f'steps {args.total_steps:,}  envs {args.num_envs}  '
          f'device {DEVICE}')
    # 2 frames per transition: the deque stores state and state_ separately.
    # Priced before allocating, since allocating to find out is the crash.
    print(f'buffer {capacity:,} transitions = '
          f'{2 * capacity * frame_bytes / 1e9:.1f} GB host RAM')
    shard_note = ('shard*/stats.jsonl' if args.num_envs > 1
                  else 'stats.jsonl')
    print(f'logging to {logdir}/{shard_note}', flush=True)

    net, buffer, extra = build(args, capacity)
    agent = agent_cls(
        net,
        buffer,
        O.Adam(net.parameters(), lr=args.lr, eps=args.adam_eps),
        # The schedule advances once per agent.step() call, i.e. once per
        # num_envs env steps, so the decay length is in calls, not env steps.
        # Without this division a 16-env run would still be at eps ~0.95 when
        # the budget runs out and would score at the random baseline.
        get_epsilon_schedule(start=args.eps_start, end=args.eps_end,
                             steps=max(1, args.eps_decay_steps
                                       // args.num_envs)),
        N_ACTIONS,
        gamma=args.gamma,
        warmup_steps=args.warmup_steps,
        update_steps=args.update_steps,
        update_target_steps=args.update_target_steps,
        num_envs=args.num_envs,
        **extra,
    )

    _, episodes, elapsed = train(agent, env, step_num=args.total_steps,
                                 log_every=args.log_every,
                                 use_mask=args.mask)

    # final_eps printed explicitly: the progress lines are gated on
    # --log-every, so a run shorter than one interval has none to scrape.
    print(f'\ndone: {episodes} episodes in {elapsed / 60:.1f} min '
          f'({args.total_steps / elapsed:.0f} fps), final_eps={agent.eps:.3f}')
    report(logdir, episodes)


if __name__ == '__main__':
    main()
