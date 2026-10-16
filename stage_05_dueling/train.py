"""Stage 05 -- dueling head, on top of stage 04's n-step Double DQN.

WHERE: local for a 2000-step smoke test, then RunPod for 200k x 2 seeds
(about 30 min, about $0.18). Never launch a pod to find out whether it runs.

WHAT: stage 04 with ONE thing changed -- QNetwork becomes DuelingQNetwork.
The trunk, the n-step buffer, the loop, the Double-DQN target, --n-step and
every other hyperparameter are stage 04 verbatim, so the 200k x 2-seed
measurement against stage 04 isolates exactly one variable: the shape of the
head. The ten lines that matter are in common/networks.py, not here.

    Wang et al. 2016, "Dueling Network Architectures for Deep Reinforcement
    Learning", https://arxiv.org/abs/1511.06581

THE PROBLEM WITH A PLAIN HEAD. Stage 04's head is one nn.Linear from 512
features to 17 action-values, and on any given transition only the action
actually taken receives a gradient -- the other 16 outputs get exactly zero.
But Crafter's states are overwhelmingly "standing on grass with nothing
nearby", where all 17 buttons are worth very nearly the same thing. The
network ends up re-deriving that single shared value seventeen separate
times, one action per visit.

THE FIX. Give the shared part its own output:

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

V(s) is a single number trained by EVERY transition regardless of which
button was pressed, and the advantage stream only has to learn the small
deviations around it. That is where the sample-efficiency comes from, and on
a 1M-step budget sample-efficiency is the whole game.

The mean subtraction is not cosmetic: without it the split V + A is not
unique (add 100 to V, subtract 100 from every A, Q is unchanged), so the two
streams drift together unboundedly. The long version of that argument, and
the code, are in common/networks.py DuelingQNetwork.

WHAT TO EXPECT: small. The guide's own verdict is "expect small, cheap enough
to keep", and the dueling paper's larger gains come from much bigger action
spaces -- 17 is not big. It costs +86% parameters (609k -> 1.14M) but almost
no wall-clock, because the conv trunk dominates the forward pass, not the
head. Crafter's binding constraint is sparse delayed reward, which is what
stage 04 addressed; this stage is a cheap architectural tidy-up on top.

DONE WHEN, at 200k on 2 seeds: return at least as good as stage 04's, no new
instability. If it comes out flat, that is a legitimate null result -- revert
it and put the revert in the slides (contribution.md section 6: three clearly
reported null results plus a working base agent beat one unexplained number).

IF IT DOES NOT LEARN, check these, most common first:
  1. the mean subtraction missing from forward() -> NO CRASH, loss still
     falls, mean-Q drifts away from any sensible scale while return stays
     flat. THAT DRIFT IS THE BUG SIGNATURE.
  2. keepdim=False on the mean -> the (B, 17) minus (B,) broadcast gives
     (B, B) or mis-centres silently. Watch the shapes, not the loss.
  3. q and q_target not the same class -> load_state_dict raises at startup.
  4. every stage 02/03/04 failure mode (permute, /255, mask from `done`,
     target net never refreshed, argmax on the wrong net, gamma**n_step
     instead of the per-transition gamma**n) still applies unchanged.

Run:
    # local smoke test, ALWAYS first, a few minutes
    python stage_05_dueling/train.py --total-steps 2000 \
        --learning-starts 500 --buffer-size 5000

    # the real run
    python stage_05_dueling/train.py --total-steps 200000 --seed 1
"""

import argparse
import pathlib
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, bootstrap_mask, make_env   # noqa: E402
from common.metrics import ACHIEVEMENTS, score_run           # noqa: E402
from common.networks import DuelingQNetwork                   # noqa: E402
from common.replay import NStepReplayBuffer, bytes_for       # noqa: E402

STAGE_ID = 'stage_05_dueling'


def parse_args(argv=None):
    """Defaults ARE the best hyperparameters, per assignment_v0.pdf 2.1:
    bare `python train.py` must start the run we actually report.

    Identical to stage 04, --n-step included. The dueling head is an
    architecture change and adds no hyperparameter, so the comparison against
    stage 04 has exactly one moving part.
    """
    p = argparse.ArgumentParser(
        description='Stage 05: n-step Double DQN + a dueling head on Crafter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--logdir', type=str, default='logdir')
    p.add_argument('--device', type=str, default=None,
                   help='cuda / cpu. Default: cuda when available')

    # --- the one new knob -------------------------------------------------
    p.add_argument('--n-step', type=int, default=10,
                   help='rewards accumulated before bootstrapping. '
                        'Sweep {3,5,10} once at 200k; DER uses 20')

    # --- replay -----------------------------------------------------------
    # NOTE the halved default against stage 03: this buffer stores next_obs
    # explicitly, so it is ~2x the bytes per frame. 500k here is 12.3 GB,
    # the same footprint stage 03 had at 1M.
    p.add_argument('--buffer-size', type=int, default=500_000,
                   help='host RAM, ~24.6GB at 1M because n-step stores '
                        'next_obs too. CAP AT 50k FOR LOCAL RUNS')
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


def push_nstep(buffer, queue, gamma):
    """Collapse the window currently in `queue` into one buffer slot.

    `queue` holds up to n_step tuples (obs, action, reward, mask, next_obs,
    done) in time order. The slot written is the one STARTING at queue[0]:
    obs and action come from the front, the return is accumulated forward, and
    next_obs is the frame the window ends at.

    THE EARLY STOP IS THE POINT. The loop breaks the first time mask == 0.0,
    which is an actual death. Everything after a death belongs to a different
    episode in a freshly generated world, so it must not enter the return, and
    the window's length k+1 becomes this slot's n. That is what makes the
    stored n per-transition rather than global.

    Kept deliberately identical to _push in common.replay.test_nstep_buffer,
    which is what verifies it. If this changes, change that too.
    """
    G, mask_n, k, nxt = 0.0, 1.0, 0, None
    for k, (_, _, r, m, nx, _) in enumerate(queue):
        G += (gamma ** k) * r
        mask_n, nxt = m, nx
        if m == 0.0:                 # died inside the window: stop here
            break
    obs0, act0 = queue[0][0], queue[0][1]
    buffer.add_nstep(obs0, act0, G, nxt, mask_n, n=k + 1,
                     done=(mask_n == 0.0))


def flush_tail(buffer, queue, gamma):
    """Push every partial window left in the queue when an episode ends.

    These are the n-1 transitions leading into the agent's own death, i.e.
    the most informative ones in the episode. Dropping them is a silent bug:
    training still works, the agent just never learns what kills it.

    The front window is pushed here only if it has NOT already been pushed by
    the full-window path -- that is the `len(queue) < n_step` test.
    """
    if not queue:
        return
    # queue.maxlen IS n_step, so this asks "was this window short?". A full
    # window already reached the buffer via the push in the main loop;
    # pushing it again here would double-count it.
    if len(queue) < queue.maxlen:
        push_nstep(buffer, queue, gamma)
    while len(queue) > 1:
        queue.popleft()
        push_nstep(buffer, queue, gamma)
    queue.clear()


def learn(q, q_target, opt, buffer, rng, args, device):
    """One gradient step on the n-step Bellman error. Returns (loss, mean Q).

    IDENTICAL TO STAGE 04, character for character. The dueling split lives
    entirely inside the network's forward(), so the learning step cannot tell
    the two architectures apart -- which is precisely why the 05-vs-04 row
    isolates the head and nothing else.
    """
    obs, act, ret, next_obs, mask, n = buffer.sample(args.batch_size, rng)

    obs = torch.as_tensor(obs, device=device)
    next_obs = torch.as_tensor(next_obs, device=device)
    act = torch.as_tensor(act, device=device)
    ret = torch.as_tensor(ret, device=device)
    mask = torch.as_tensor(mask, device=device)
    # float, not int: it goes into an exponent that must stay differentiable-
    # shaped and broadcast against float tensors.
    n = torch.as_tensor(n, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Double DQN, stage 03 verbatim: online CHOOSES, frozen target SCORES.
        best = q(next_obs).argmax(dim=1, keepdim=True)            # (B, 1)
        next_q = q_target(next_obs).gather(1, best).squeeze(1)    # (B,)
        # gamma ** n, NOT gamma ** args.n_step. The whole stage turns on this
        # line: a window cut short by death carries a smaller exponent.
        target = ret + (args.gamma ** n) * mask * next_q

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

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.seed, logdir)

    # THE ONLY LINES THAT DIFFER FROM STAGE 04. Same trunk, same n-step
    # Double-DQN target, same buffer, same hyperparameters -- only the head
    # splits. Both nets must be the SAME class or load_state_dict below
    # raises, which is the one failure mode here that is honest about itself.
    q = DuelingQNetwork(N_ACTIONS).to(device)
    q_target = DuelingQNetwork(N_ACTIONS).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=args.lr, eps=args.adam_eps)

    capacity = min(args.buffer_size, args.total_steps)
    # Price it BEFORE allocating: on a box that cannot hold the buffer,
    # allocating it to find out is itself the crash.
    planned = bytes_for(capacity, nstep=True)
    buffer = NStepReplayBuffer(capacity)

    # The window accumulator. maxlen is what makes a full window detectable.
    queue = deque(maxlen=args.n_step)

    print(f'stage {STAGE_ID}  seed {args.seed}  steps {args.total_steps:,}  '
          f'device {device}')
    print(f'n_step {args.n_step}  gamma {args.gamma}  '
          f'effective horizon ~{1 / (1 - args.gamma):.0f} steps')
    print(f'buffer {buffer.capacity:,} frames = {buffer.nbytes() / 1e9:.1f} GB '
          f'host RAM (planned {planned / 1e9:.1f} GB, 2x a 1-step buffer)')
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
                values = q(torch.as_tensor(obs[None], device=device))
                action = int(values.argmax())

        next_obs, reward, done, info = env.step(action)
        mask = bootstrap_mask(done, info)

        # --- store --------------------------------------------------------
        # Into the WINDOW, not the buffer. A slot is only written once n
        # rewards are known (or the episode ends and the tail is flushed).
        queue.append((obs, action, reward, mask, next_obs, done))
        if len(queue) == args.n_step:
            push_nstep(buffer, queue, args.gamma)

        ep_return += reward
        ep_len += 1
        obs = next_obs

        if done:
            # Flush BEFORE reset, while the queue still holds this episode's
            # final steps. After reset those frames belong to a new world.
            flush_tail(buffer, queue, args.gamma)
            episodes += 1
            returns.append(ep_return)
            obs = env.reset()
            ep_return, ep_len = 0.0, 0

        # --- learn --------------------------------------------------------
        if (step >= args.learning_starts and len(buffer) > args.batch_size
                and step % args.train_every == 0):
            loss, mean_q = learn(q, q_target, opt, buffer, rng, args, device)

        if step % args.target_update == 0:
            q_target.load_state_dict(q.state_dict())

        # --- log ----------------------------------------------------------
        if (step + 1) % args.log_every == 0:
            fps = (step + 1) / (time.time() - start)
            recent = np.mean(returns[-20:]) if returns else float('nan')
            # slots/step catches a missing tail flush: it should sit just
            # under 1.0 (only the in-flight window is unwritten).
            print(f'step {step + 1:>8,}  ep {episodes:>4}  '
                  f'return {recent:6.2f}  eps {eps:4.2f}  '
                  f'loss {loss:8.4f}  meanQ {mean_q:7.3f}  '
                  f'slots/step {len(buffer) / (step + 1):4.2f}  {fps:>5.0f} fps')

    elapsed = time.time() - start
    print(f'\ndone: {episodes} episodes in {elapsed / 60:.1f} min '
          f'({args.total_steps / elapsed:.0f} fps)')

    stats = logdir / 'stats.jsonl'
    if stats.exists() and episodes:
        score, percents, n_ep = score_run(stats)
        print(f'score over {n_ep} episodes: {score:.2f}%   '
              f'(random baseline: ~1.6%)')
        print('\nper-achievement success rate, non-zero only:')
        for name, pct in sorted(zip(ACHIEVEMENTS, percents),
                                key=lambda kv: -kv[1]):
            if pct > 0:
                print(f'  {name:<20} {pct:5.1f}%')


if __name__ == '__main__':
    main()
