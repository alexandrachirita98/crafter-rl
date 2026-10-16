"""Stage 04 -- n-step returns.

WHERE: local for a 2000-step smoke test, then RunPod for 200k x 2 seeds
(about 30 min, about $0.18). Never launch a pod to find out whether it runs.

WHAT: stage 03 with the one-step buffer swapped for NStepReplayBuffer and one
term changed in the target. The network, the loop and every hyperparameter are
stage 03 verbatim, plus the single new knob --n-step.

    Sutton 1988, "Learning to Predict by the Methods of Temporal
    Differences"; Sutton & Barto 2018 ch. 7

THE PROBLEM, AND WHY IT IS THE BIGGEST ONE ON CRAFTER. Stages 02-03 bootstrap
from the very next state:

    target = r_t + gamma * mask * Q_target(s_{t+1}, .)

so reward information moves back exactly ONE step per gradient update. Crafter
pays +1 once for an achievement and then nothing for hundreds of steps. For the
agent to learn that chopping wood 300 steps ago is what made the table
possible, that one scalar has to crawl back 300 bootstrap links, each one
attenuated by a noisy function approximator. Within a 1M-step budget it does
not arrive.

THE FIX. Use the next n REAL rewards before bootstrapping at all:

    G_t = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
    target = G_t + gamma^n * mask * Q_target(s_{t+n}, .)

Information now travels n steps per update: at n = 10, ten times faster. The
cost is bias -- the rewards come from the behaviour policy, not the greedy one
-- which is why n is a sweep and not a constant. DER uses 20; we start at 10.
This is also exactly the quantity CONTRIBUTION 2 later makes adaptive.

THE CLASSIC n-STEP BUG, WHICH THIS STAGE EXISTS TO GET RIGHT. If the episode
ended 4 steps into a 10-step window, that transition accumulated 4 rewards, so
its bootstrap discount is gamma^4, not gamma^10. Using one global n:

    target = G + (args.gamma ** args.n_step) * mask * next_q   # WRONG

mis-discounts every window near a death -- and on Crafter deaths are exactly
where the reward signal lives. The buffer therefore stores n PER TRANSITION and
the target reads gamma^n with the n that sample() returned. Nothing crashes if
you get this wrong; the agent just quietly undervalues everything near a death.

WHAT ELSE CHANGES, AND WHY IT HAS TO. The one-step buffer saved 12 GB by
exploiting next_obs(t) == obs(t+1). That identity is false here: the successor
of an n-step window is the frame n steps later, and for a window cut short by
death it is not at a fixed offset. So NStepReplayBuffer stores next_obs
explicitly and costs 2x the RAM -- 24.6 GB at 1M frames, which is why
--buffer-size matters more from this stage onward. Check with
common.replay.bytes_for(capacity, nstep=True) BEFORE launching.

THE TAIL FLUSH, which is easy to drop and loses the most important
transitions. When an episode ends, the accumulator still holds up to n-1
partial windows. Every one of them is a real transition, and they are the n-1
steps that led directly to the agent's own death. Drop them and the agent never
sees what killed it.

WHAT TO EXPECT: the big one. The guide's stage table says **big**, and it is
the first stage where return should visibly move rather than merely stabilise.

DONE WHEN, at 200k on 2 seeds: clearly better return than stage 03, and
collect_wood / place_table showing up at a higher rate than stage 03's.

IF IT DOES NOT LEARN, check these, most common first:
  1. gamma ** args.n_step instead of gamma ** n   -> the classic bug above.
     Looks fine, trains fine, systematically wrong near every death.
  2. tail flush missing      -> len(buffer) comes out noticeably below the
     step count. The printed ratio at each log line catches this.
  3. next_obs read at i+1    -> NStepReplayBuffer.sample() returns next_obs
     explicitly; using obs[idx+1] gives the 1-step successor with an n-step
     return, which is a target that mixes two horizons.
  4. the stage 02/03 failure modes (permute, /255, mask from `done`, target
     net never refreshed, argmax on the target net) all still apply.

Run:
    # local smoke test, ALWAYS first, about 3 minutes
    python stage_04_nstep/train.py --total-steps 2000 \
        --learning-starts 500 --buffer-size 5000

    # the real run
    python stage_04_nstep/train.py --total-steps 200000 --seed 1
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
from common.networks import QNetwork                         # noqa: E402
from common.replay import NStepReplayBuffer, bytes_for       # noqa: E402

STAGE_ID = 'stage_04_nstep'


def parse_args(argv=None):
    """Defaults ARE the best hyperparameters, per assignment_v0.pdf 2.1:
    bare `python train.py` must start the run we actually report.

    Identical to stage 03 except for --n-step, so the comparison between the
    two stages has exactly one moving part.
    """
    p = argparse.ArgumentParser(
        description='Stage 04: Double DQN + n-step returns on Crafter.',
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
    G, mask_n, k, nxt, end = 0.0, 1.0, 0, None, False
    for k, (_, _, r, m, nx, d) in enumerate(queue):
        G += (gamma ** k) * r
        mask_n, nxt, end = m, nx, d
        if m == 0.0:                 # died inside the window: stop here
            break
    obs0, act0 = queue[0][0], queue[0][1]
    # `done` is the episode boundary -- death OR the 10,000-step timeout --
    # and is NOT the same as mask_n == 0.0, which is death alone. A timeout
    # ends the episode and still bootstraps. Stored for later stages; this
    # stage's sample() never reads it.
    buffer.add_nstep(obs0, act0, G, nxt, mask_n, n=k + 1, done=end)


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

    THE ONLY DIFFERENCES FROM STAGE 03: sample() hands back an extra `n`, and
    the bootstrap discount is gamma**n with that PER-TRANSITION n. Double DQN
    is untouched, so the 04-vs-03 row isolates the n-step return alone.
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

    q = QNetwork(N_ACTIONS).to(device)
    q_target = QNetwork(N_ACTIONS).to(device)
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
