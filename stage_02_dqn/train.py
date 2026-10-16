"""Stage 02 -- plain DQN.

WHERE: local for a 2000-step smoke test, then RunPod for 200k x 2 seeds
(about 30 min, about $0.18). Never launch a pod to find out whether it runs.

WHAT: Mnih et al. 2015 (Nature 518:529-533), optimised with Adam. A CNN maps
the 64x64x3 frame to 17 action-values; we act by argmax with epsilon-greedy
exploration and learn by regressing onto the Bellman target

    target = r + gamma * mask * max_a' Q_target(s', a')

WHY THIS IS STAGE 2, before any of the clever parts:

  "Start small. Implement an algorithm you already have some experience with,
   such as DQN, optimized with Adam, and only after you convince yourself it's
   better than random start modding it."   -- assignment_v0.pdf section 2.3

Two pieces of machinery, both from the Nature paper, and the run diverges
without either:

  REPLAY BUFFER  consecutive Crafter frames are nearly identical, so learning
                 from them in order makes the network chase its own tail.
                 common/replay.py.
  TARGET NETWORK a frozen copy of Q, refreshed every --target-update steps.
                 Without it we fit a guess to a number computed from that same
                 guess. This was the paper's main stability contribution.

DONE WHEN, at 200k on 2 seeds: mean return clearly above random's ~2.1; the
agent reliably gets collect_wood, collect_drink, eat_cow; loss falling and
mean-Q rising then flattening rather than exploding.

IF IT DOES NOT LEARN, check these four, most common first:
  1. permute missing          -> convolving the wrong axes (common/networks.py)
  2. /255 missing             -> saturated ReLUs (common/networks.py)
  3. mask taken from `done`   -> poisoned timeouts (common/env.bootstrap_mask)
  4. target net never updated, or updated every step -> no target, or no
     stability

Run:
    # local smoke test, ALWAYS first, about 3 minutes
    python stage_02_dqn/train.py --total-steps 2000 \
        --learning-starts 500 --buffer-size 5000

    # the real run
    python stage_02_dqn/train.py --total-steps 200000 --seed 1
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, bootstrap_mask, make_env   # noqa: E402
from common.metrics import ACHIEVEMENTS, score_run           # noqa: E402
from common.networks import QNetwork                         # noqa: E402
from common.replay import ReplayBuffer                       # noqa: E402

STAGE_ID = 'stage_02_dqn'


def parse_args(argv=None):
    """Defaults ARE the best hyperparameters, per assignment_v0.pdf 2.1:
    bare `python train.py` must start the run we actually report.

    Values from CleanRL and van Hasselt et al. 2019 (Data-Efficient Rainbow),
    adapted to a 1M budget. These carry forward through every later stage.
    """
    p = argparse.ArgumentParser(
        description='Stage 02: plain DQN on Crafter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--logdir', type=str, default='logdir')
    p.add_argument('--device', type=str, default=None,
                   help='cuda / cpu. Default: cuda when available')

    # --- replay -----------------------------------------------------------
    p.add_argument('--buffer-size', type=int, default=1_000_000,
                   help='host RAM, ~12GB at 1M. CAP AT 100k FOR LOCAL RUNS')
    p.add_argument('--learning-starts', type=int, default=20_000,
                   help='fill the buffer before training on it')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--train-every', type=int, default=1,
                   help='gradient steps per env step; sweep {1,2,4} once at 200k')

    # --- optimisation -----------------------------------------------------
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Rainbow uses 6.25e-5; 1e-4 learns faster on a short budget')
    p.add_argument('--adam-eps', type=float, default=1.5e-4,
                   help="the RL-specific value, NOT torch's 1e-8")
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--target-update', type=int, default=2000,
                   help="DER value; faster than Atari's 8000, the budget is small")

    # --- exploration ------------------------------------------------------
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-steps', type=int, default=250_000,
                   help='linear over the first 25%% of a 1M run')

    p.add_argument('--log-every', type=int, default=10_000)
    return p.parse_args(argv)


def epsilon(step, args):
    """Linear decay from --eps-start to --eps-end, then flat."""
    frac = min(1.0, step / max(1, args.eps_decay_steps))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def learn(q, q_target, opt, buffer, rng, args, device):
    """One gradient step on the Bellman error. Returns (loss, mean Q)."""
    obs, act, rew, next_obs, mask = buffer.sample(args.batch_size, rng)

    obs = torch.as_tensor(obs, device=device)
    next_obs = torch.as_tensor(next_obs, device=device)
    act = torch.as_tensor(act, device=device)
    rew = torch.as_tensor(rew, device=device)
    mask = torch.as_tensor(mask, device=device)

    with torch.no_grad():
        # The frozen copy supplies the target. Using q here instead of
        # q_target is failure mode 4 in the module docstring.
        next_q = q_target(next_obs).max(dim=1).values
        # mask is info['discount']: 0.0 ONLY on death. A 10,000-step timeout
        # still bootstraps. See common/env.py, THE EPISODE-END TRAP.
        target = rew + args.gamma * mask * next_q

    pred = q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(pred, target)    # Huber: robust to outliers

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
    opt.step()
    return loss.item(), pred.mean().item()


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    logdir = (pathlib.Path(args.logdir) / f'crafter_reward-{STAGE_ID}'
              / str(args.seed))
    logdir.mkdir(parents=True, exist_ok=True)

    # Seed everything that draws: torch for the init and any GPU op, one numpy
    # Generator for both the policy's coin flips and the buffer's sampling.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.seed, logdir)

    q = QNetwork(N_ACTIONS).to(device)
    q_target = QNetwork(N_ACTIONS).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=args.lr, eps=args.adam_eps)

    buffer = ReplayBuffer(min(args.buffer_size, args.total_steps))

    print(f'stage {STAGE_ID}  seed {args.seed}  steps {args.total_steps:,}  '
          f'device {device}')
    print(f'buffer {buffer.capacity:,} frames = {buffer.nbytes() / 1e9:.1f} GB '
          f'host RAM')
    print(f'logging to {logdir}/stats.jsonl')

    obs = env.reset()
    ep_return, ep_len, episodes = 0.0, 0, 0
    loss, mean_q = float('nan'), float('nan')
    returns, start = [], time.time()

    for step in range(args.total_steps):
        # --- act ----------------------------------------------------------
        eps = epsilon(step, args)
        if rng.random() < eps:
            action = int(rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                # obs[None] adds the batch axis the conv stack expects.
                values = q(torch.as_tensor(obs[None], device=device))
                action = int(values.argmax())

        next_obs, reward, done, info = env.step(action)

        # --- store --------------------------------------------------------
        # bootstrap_mask, never `0.0 if done else 1.0`: the mask must stay 1.0
        # when the episode merely hit the 10,000-step cap.
        buffer.add(obs, action, reward, bootstrap_mask(done, info), done)

        ep_return += reward
        ep_len += 1
        obs = next_obs

        if done:
            episodes += 1
            returns.append(ep_return)
            obs = env.reset()
            ep_return, ep_len = 0.0, 0

        # --- learn --------------------------------------------------------
        if step >= args.learning_starts and step % args.train_every == 0:
            loss, mean_q = learn(q, q_target, opt, buffer, rng, args, device)

        # Hard copy of the weights, not a Polyak average: the Nature recipe.
        if step % args.target_update == 0:
            q_target.load_state_dict(q.state_dict())

        # --- log ----------------------------------------------------------
        if (step + 1) % args.log_every == 0:
            fps = (step + 1) / (time.time() - start)
            recent = np.mean(returns[-20:]) if returns else float('nan')
            print(f'step {step + 1:>8,}  ep {episodes:>4}  '
                  f'return {recent:6.2f}  eps {eps:4.2f}  '
                  f'loss {loss:8.4f}  meanQ {mean_q:7.3f}  {fps:>5.0f} fps')

    elapsed = time.time() - start
    print(f'\ndone: {episodes} episodes in {elapsed / 60:.1f} min '
          f'({args.total_steps / elapsed:.0f} fps)')

    # Same closing report as stage 01, so the two are read side by side.
    stats = logdir / 'stats.jsonl'
    if stats.exists() and episodes:
        score, percents, n = score_run(stats)
        print(f'score over {n} episodes: {score:.2f}%   '
              f'(random baseline: ~1.6%)')
        print('\nper-achievement success rate, non-zero only:')
        for name, pct in sorted(zip(ACHIEVEMENTS, percents),
                                key=lambda kv: -kv[1]):
            if pct > 0:
                print(f'  {name:<20} {pct:5.1f}%')


if __name__ == '__main__':
    main()
