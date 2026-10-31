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

The biased max target is the lab's and is left alone -- stage 03 replaces it.
MSE and no clipping were the lab's too, and are still reachable with
--loss mse --clip-grad 0, which reproduces every run recorded before the
flags existed; the defaults are now Huber and a norm clip of 10.

Run:
    python stage_02_dqn/train.py --total-steps 200000 --seed 1
"""

import argparse
import itertools
import json
import pathlib
import random
import signal
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
from common.metrics import (ACHIEVEMENTS, compute_scores,
                            score_pooled, success_percents)

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
        """-> states, actions, rewards, states_, mask, legal_. Frames uint8.

        legal_ is the (B,17) legal-action mask of the NEXT state, for masking
        the bootstrap. It is all-ones when the run is unmasked, which makes
        the masked term a no-op and keeps an unmasked run bit-identical.
        """
        s, a, r, s_, m, lg = zip(
            *random.sample(self._buffer, self._batch_size))
        return (
            _to_tensor(np.stack(s)),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            _to_tensor(np.stack(s_)),
            torch.tensor(m, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.from_numpy(np.stack(lg)).to(DEVICE),
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
        loss_kind='huber',
        clip_grad=10.0,
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
        self._loss_kind = loss_kind
        self._clip_grad = clip_grad
        # state_'s legal mask for the batch being updated. The quantile
        # stages take their argmax inside _target_quantiles, several
        # overrides deep, and threading an extra positional through all of
        # them is how a stage ends up masking with the WRONG batch's mask.
        # _update sets this; the argmax sites read it. None means all-ones.
        self._legal_ = None
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

    def _q(self, states):
        """Q-values, (B, A). Stage 06's quantile head overrides this to take
        the mean over quantiles, so act_greedy works there unchanged.
        """
        return self._estimator(states)

    def act_greedy(self, states, masks=None):
        """Argmax over legal actions, no epsilon and no warmup.

        Separate from step() on purpose: step() advances the epsilon
        schedule and the warmup counter, so calling it for evaluation would
        consume the training schedule and make eval change training.
        """
        n = len(states)
        masks = _mask_or_ones(masks, n, self._action_num)
        with torch.no_grad():
            qvals = self._q(_to_tensor(np.asarray(states)))
            qvals = _apply_mask(qvals, masks)
            return qvals.argmax(1).cpu().numpy()

    def learn(self, i, state, action, reward, state_, mask, legal_=None):
        """`i` is which env the transition came from. Stages that keep
        per-trajectory state (04's n-step window, 09's repeat walk) index by
        it; the replay buffer itself does not care.

        legal_ is state_'s legal-action mask, stored so _update can mask the
        bootstrap. None means unmasked and becomes all-ones.
        """
        if legal_ is None:
            legal_ = np.ones(self._action_num, dtype=np.float32)
        self._store(i, state, action, reward, state_, mask, legal_)

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

    def _store(self, i, state, action, reward, state_, mask, legal_):
        """One-step: push as-is. Stage 04 keeps an n-step window here."""
        del i
        self._buffer.push((state, action, reward, state_, mask, legal_))

    def end_episode(self, i):
        """train() calls this when env `i` finishes. One-step does nothing."""
        del i

    def _mask_next(self, qvals):
        """Mask Q-values at state_ with the stashed next-state legal set.

        getattr, not self._legal_: the stages' self-tests build agents with
        __new__ and set only the few attributes they exercise, so the stash
        can legitimately not exist yet.
        """
        legal_ = getattr(self, '_legal_', None)
        if legal_ is None:
            return qvals
        return _apply_mask(qvals, legal_)

    def _td_loss(self, qsa, target_qsa):
        """The regression loss, as chosen by --loss.

        mse is the lab's and reproduces every run recorded before the flag
        existed. huber is quadratic under 1 and linear above, so an
        achievement's +1-after-hundreds-of-zeros cannot scale the gradient
        with the error the way a square does.
        """
        if self._loss_kind == 'mse':
            return (qsa - target_qsa).pow(2).mean()
        return nn.functional.smooth_l1_loss(qsa, target_qsa)

    def _backward(self, loss, zero_first=True):
        """One gradient step, clipped. Every stage's _update ends here.

        The clip is on the whole-net norm, not per parameter: the conv trunk
        is shared by all 17 action-values, so one spiked action-value is what
        damages the features the other 16 read. 0 disables it and reproduces
        the pre-flag runs.

        zero_first=False is for the stages that zero the gradient AFTER
        stepping (11, 12); the clip has to land between backward and step
        either way, which is the whole reason this is one method.
        """
        if zero_first:
            self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad:
            nn.utils.clip_grad_norm_(self._estimator.parameters(),
                                     self._clip_grad)
        self._optimizer.step()
        if not zero_first:
            self._optimizer.zero_grad()

    def _update(self, states, actions, rewards, states_, mask, legal_):
        """Stage 03 overrides this method and nothing else."""
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        qsa = q_values.gather(1, actions)
        # Mask before the max. Without this the target maximises over actions
        # the masked policy can never take, so their Q-values are never
        # corrected by experience and nothing stops them drifting up -- the
        # max then tracks whichever unreachable action drifted highest.
        # All-ones on an unmasked run, so this is a no-op there.
        qsa_ = _apply_mask(q_values_, legal_).max(1, keepdim=True)[0]

        # mask is info['discount']: 0.0 only on death, so a timeout still
        # bootstraps. The lab's (1 - done) would zero it.
        target_qsa = rewards + self._gamma * qsa_ * mask

        loss = self._td_loss(qsa, target_qsa)
        self._backward(loss)

        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


# ----------------------------------------------------------------- 5. loop


def save_ckpt(path, agent, step, episodes):
    """Model, target, optimizer and schedule position. Not the buffer.

    The buffer is 12 GB at a 500k capacity, so writing it every interval
    would cost more wall clock than the training it protects. That makes this
    a warm start, not an exact resume: weights come back, the replay
    distribution does not. Enough to rescue a run the node killed; not enough
    to call the result a continuous run of the combined length.

    Writes to a temp path and renames. A SIGKILL partway through torch.save
    leaves a truncated file, and overwriting in place would destroy the last
    good checkpoint to produce it.
    """
    tmp = path.with_suffix('.tmp')
    torch.save({
        'step': step,
        'episodes': episodes,
        'net': agent._estimator.state_dict(),
        'target': agent._target_estimator.state_dict(),
        'optim': agent._optimizer.state_dict(),
        'step_cnt': agent._step_cnt,
        'eps': agent.eps,
    }, tmp)
    tmp.replace(path)


def load_ckpt(path, agent):
    """-> (step, episodes) to resume from. Mutates agent in place.

    The epsilon schedule is a generator and cannot be pickled, so it is
    fast-forwarded by consuming it step_cnt times rather than restored.
    """
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    agent._estimator.load_state_dict(ck['net'])
    agent._target_estimator.load_state_dict(ck['target'])
    agent._optimizer.load_state_dict(ck['optim'])
    agent.eps = ck['eps']

    # The buffer is not in the checkpoint, so it comes back empty. learn()
    # gates on step_cnt < warmup_steps, and restoring step_cnt as-is would put
    # the agent straight past the gate and sample a batch out of an empty
    # deque. Rewind step_cnt so the first warmup_steps calls only store, then
    # let it carry on from where it was.
    agent._step_cnt = 0
    # Fast-forward the schedule so exploration resumes where it stopped
    # rather than restarting at eps=1.0. The generator cannot be pickled, so
    # it is advanced rather than restored.
    for _ in range(ck['step_cnt']):
        next(agent._epsilon, None)
    return ck['step'], ck['episodes']


def evaluate(agent, num_envs, seed, episodes_wanted=16,
             use_mask=False, mask_extended=False, mask_skip_grass=False):
    """Run the greedy policy (eps=0) on fresh worlds and score it.

    Its own VecEnv with logdir=None, so nothing lands in the training
    stats.jsonl -- the training score has to stay the number every run in
    */results/ already reports. Seeds come from a block well above the
    training seeds so eval worlds are never training worlds.

    Returns (score_percent, mean_reward, mean_length, n_episodes).
    """
    env = VecEnv(num_envs, logdir=None, seed=10_000 + seed,
                 seed_stride=max(1, num_envs))
    states = env.reset()
    masks = None
    rewards_out, lengths_out, rows = [], [], []
    ep_reward = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    # Achievements come off info, since no Recorder is writing stats here.
    unlocked = [dict() for _ in range(num_envs)]

    while len(rows) < episodes_wanted:
        actions = agent.act_greedy(states, masks)
        states, rew, dones, infos = env.step(actions)
        if use_mask:
            masks = legal_masks(infos, dones, extended=mask_extended,
                                skip_grass=mask_skip_grass)
        for i in range(num_envs):
            ep_reward[i] += rew[i]
            ep_len[i] += 1
            ach = infos[i].get('achievements')
            if ach:
                unlocked[i] = dict(ach)
            if dones[i]:
                row = {f'achievement_{k}': int(v)
                       for k, v in unlocked[i].items()}
                row['reward'] = float(ep_reward[i])
                row['length'] = int(ep_len[i])
                rows.append(row)
                rewards_out.append(float(ep_reward[i]))
                lengths_out.append(int(ep_len[i]))
                ep_reward[i], ep_len[i] = 0.0, 0
                unlocked[i] = {}

    percents = success_percents(rows)
    score = float(compute_scores(percents))
    return (score, float(np.mean(rewards_out)), float(np.mean(lengths_out)),
            len(rows))


def train(agent, env, step_num=100_000, log_every=10_000,
          use_mask=False, mask_extended=False, mask_skip_grass=False,
          ckpt_path=None, ckpt_every=0,
          start_step=0, start_episodes=0,
          eval_every=0, eval_episodes=16, eval_seed=1, eval_log=None):
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

    mask_extended adds the `do`, `sleep` and movement rules (stage 14);
    mask_skip_grass also forbids `do` at grass. Both off by default, so
    --mask alone reproduces 1m_fix_mask.
    """
    num_envs = env.num_envs
    ep_reward = np.zeros(num_envs, dtype=np.float64)
    recent = deque(maxlen=10)
    states = env.reset()
    episodes = start_episodes
    start = time.time()

    # SLURM sends SIGTERM before the hard kill (see --signal in jobs/train.sh).
    # Catching it to checkpoint turns a timeout into a resumable run; the
    # episodes already in stats.jsonl are flushed per episode and survive
    # regardless.
    dying = {'now': False}

    def _on_term(signum, frame):
        del signum, frame
        dying['now'] = True

    if ckpt_path is not None:
        signal.signal(signal.SIGTERM, _on_term)
    # reset() returns no info, so the first step runs unmasked -- which is
    # what legal_masks would give for a fresh player anyway.
    masks = None

    done_steps = start_step
    for step in range(start_step, step_num, num_envs):

        actions = agent.step(states, masks)
        # Every stage's step() is (N,64,64,3) in, (N,) out. A stage written
        # for one env returns one action and the learn() loop below dies on
        # actions[1] fifteen frames away from the cause, so name it here.
        assert np.shape(actions) == (num_envs,), (
            f'{type(agent).__name__}.step() returned {np.shape(actions)}, '
            f'want ({num_envs},) -- a single-env step() on a vec env')

        states_, rewards, dones, infos = env.step(actions)

        if use_mask:
            # dones matters. VecEnv auto-resets and hands back the DYING
            # step's info beside the NEXT episode's first observation, so
            # masking the new state with it would forbid legal actions for a
            # whole episode's first step. legal_masks falls back to all-ones
            # wherever done is set.
            masks = legal_masks(infos, dones, extended=mask_extended,
                                skip_grass=mask_skip_grass)

        # The same masks go to the learner as state_'s legal set, and to the
        # next agent.step() as the current state's -- state_ IS the next
        # state. None when unmasked, which learn() reads as all-ones.
        for i in range(num_envs):
            # On a TIMEOUT the mask is 1.0, so the target bootstraps off
            # state_ -- and VecEnv has already replaced state_ with the next
            # world's first frame. info['final_obs'] is the real one. Costs
            # nothing today (no logged episode has ever reached the 10,000
            # cap) and stops being silently wrong when one does.
            next_state = states_[i]
            if dones[i] and 'final_obs' in infos[i]:
                next_state = infos[i]['final_obs']
            agent.learn(i, states[i], actions[i], rewards[i], next_state,
                        bootstrap_mask(dones[i], infos[i]),
                        masks[i] if use_mask else None)
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

        if ckpt_path is not None and ckpt_every \
                and done_steps % ckpt_every < num_envs:
            save_ckpt(ckpt_path, agent, done_steps, episodes)

        if eval_every and done_steps % eval_every < num_envs:
            ev = evaluate(agent, num_envs, eval_seed,
                          episodes_wanted=eval_episodes, use_mask=use_mask,
                          mask_extended=mask_extended,
                          mask_skip_grass=mask_skip_grass)
            score, mrew, mlen, n_ep = ev
            print(f'  eval @{done_steps:8d}: score={score:5.2f}%  '
                  f'R/ep={mrew:6.2f}  len={mlen:5.1f}  ({n_ep} ep, greedy)',
                  flush=True)
            if eval_log is not None:
                with open(eval_log, 'a') as fh:
                    fh.write(json.dumps({
                        'step': done_steps, 'score_percent': score,
                        'mean_episode_reward': mrew,
                        'mean_episode_length': mlen,
                        'episodes': n_ep}) + '\n')

        # A caught SIGTERM means the allocation is about to end. Save and
        # leave by the normal path so the caller still writes its summary
        # over the episodes recorded so far.
        if dying['now']:
            if ckpt_path is not None:
                save_ckpt(ckpt_path, agent, done_steps, episodes)
            print(f'\nSIGTERM at {done_steps:,} steps -- checkpointed',
                  flush=True)
            break

    return done_steps, episodes, time.time() - start


# ------------------------------------------------------------------ 6. main


def parse_args(argv=None, extra_args=None):
    p = argparse.ArgumentParser(
        description='Stage 02: the course lab DQN, on Crafter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--seed-stride', type=int, default=1,
                   help='env i gets seed*stride+i. 1 reproduces every run '
                        'already logged, but makes --seed 1 and --seed 2 '
                        'share 15 of 16 worlds at 16 envs; pass 16 (or '
                        'num-envs) for disjoint seeds')
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
    p.add_argument('--loss', dest='loss_kind', choices=('huber', 'mse'),
                   default='huber',
                   help="mse is the lab's and the pre-fix default; huber "
                        'bounds the gradient of a sparse-reward spike')
    p.add_argument('--clip-grad', type=float, default=10.0,
                   help='max gradient norm; 0 disables, which is what the '
                        'lab had')
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

    # Greedy (eps=0) evaluation on held-out worlds, for the eval curves the
    # assignment asks for. 0 disables, which is what every run already in
    # */results/ did -- their summaries are training-time numbers only.
    p.add_argument('--eval-every', type=int, default=0,
                   help='greedy-eval every N env steps; 0 disables')
    p.add_argument('--eval-episodes', type=int, default=16,
                   help='episodes per greedy eval')

    # Weights only, not the buffer: see save_ckpt. 0 disables, which is what
    # every run already in */results/ did.
    p.add_argument('--ckpt-every', type=int, default=0,
                   help='write a weights checkpoint every N env steps; '
                        '0 disables. A caught SIGTERM always writes one.')
    p.add_argument('--resume', action='store_true',
                   help='start from the checkpoint in the logdir if one is '
                        'there. Warm start, not an exact resume.')

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
    env = VecEnv(args.num_envs, logdir=logdir, seed=args.seed,
                 seed_stride=args.seed_stride)

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
        loss_kind=args.loss_kind,
        clip_grad=args.clip_grad,
        **extra,
    )

    ckpt_path = logdir / 'ckpt.pt' if args.ckpt_every or args.resume else None
    start_step, start_episodes = 0, 0
    if args.resume and ckpt_path is not None and ckpt_path.exists():
        start_step, start_episodes = load_ckpt(ckpt_path, agent)
        print(f'resuming from {ckpt_path} at {start_step:,} steps '
              f'(warm start: buffer is empty)', flush=True)

    ran_steps, episodes, elapsed = train(
        agent, env, step_num=args.total_steps, log_every=args.log_every,
        use_mask=args.mask,
        mask_extended=getattr(args, 'mask_extended', False),
        mask_skip_grass=getattr(args, 'mask_skip_grass', False),
        ckpt_path=ckpt_path, ckpt_every=args.ckpt_every,
        start_step=start_step, start_episodes=start_episodes,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        eval_seed=args.seed, eval_log=logdir / 'eval.jsonl')

    # final_eps printed explicitly: the progress lines are gated on
    # --log-every, so a run shorter than one interval has none to scrape.
    # ran_steps, not --total-steps: a SIGTERM break stops early and exits 0,
    # so this line is the only place the real length is recorded. run.sh reads
    # it back to mark the summary partial.
    print(f'\ndone: {episodes} episodes in {elapsed / 60:.1f} min '
          f'({ran_steps / elapsed:.0f} fps), final_eps={agent.eps:.3f}, '
          f'ran_steps={ran_steps}, requested_steps={args.total_steps}')
    report(logdir, episodes)


if __name__ == '__main__':
    main()
