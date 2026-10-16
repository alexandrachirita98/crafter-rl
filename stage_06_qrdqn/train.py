"""Stage 06 -- QR-DQN, 51-quantile distributional learning.

WHERE: 💻 SMOKE-TEST THE LOSS ITSELF LOCALLY FIRST -- `python
stage_06_qrdqn/train.py --test-loss` -- then the 2000-step run, then RunPod for
200k x 2 seeds (about 30 min, about $0.18). This is the one stage where a
numerical unit test on the loss is worth more than the smoke test: the loss can
be subtly wrong, train happily, and simply learn less.

WHAT: stage 05 with the head widened to N quantiles per action and the Huber
loss replaced by the QUANTILE Huber loss. The trunk, the buffer, the n-step
accumulator, Double DQN and the loop are stage 05 verbatim.

    Dabney, Rowland, Bellemare & Munos 2018, "Distributional Reinforcement
    Learning with Quantile Regression", https://arxiv.org/abs/1710.10044,
    section 3 and appendix A. READ IT PROPERLY FOR THIS ONE.

THE IDEA. Stages 02-05 predict one number per action: the EXPECTED return.
This predicts N numbers describing the whole SPREAD of returns -- "this action
usually gives 3 but sometimes 12". The agent still acts on the mean, so the
policy is unchanged in form; what changes is how much the network is asked to
predict from each transition.

WHY IT IS THE BIGGEST WIN ON A SMALL BUDGET. One transition used to supply one
regression target. It now supplies N, each pinned to a different part of the
distribution, so the trunk gets N times the gradient signal from the same 1M
environment steps. contribution.md section 2.1 notes that EDE -- the only
DQN-family agent that does well on Crafter, at 11.7% -- makes exactly this swap
as its FIRST change to Rainbow.

WHY QUANTILES AND NOT C51. C51 fixes a support [V_min, V_max] up front and
projects onto it. On Crafter nobody knows the right range and a bad guess
silently clips the distribution. QR-DQN learns the LOCATIONS and fixes the
PROBABILITIES at 1/N instead: no range to guess, no projection step. That is
why contribution.md section 5 rejects C51.

THE THREE PLACES THIS GOES WRONG. All three run without raising, and all three
just make the agent learn less -- which is why --test-loss exists.

  1. taus must be MIDPOINTS (2i+1)/(2N), not edges i/N. With i/N the first
     tau is 0, whose loss weight |tau - 1{u<0}| is zero for every
     over-estimate, so that head gets no gradient in one direction.
     common.networks.quantile_midpoints is the only place this is computed,
     and `python -m common.networks` asserts it.
  2. BROADCASTING DIRECTION. `target.unsqueeze(-1) - pred.unsqueeze(-2)` gives
     (B, N_target, N_pred). Reverse it and the loss is transposed -- still a
     scalar, still decreasing, and wrong.
  3. WHICH AXIS IS SUMMED VS AVERAGED. Sum over TARGET quantiles, mean over
     PREDICTION quantiles. Swapping them rescales the gradient by N, which
     looks like a learning-rate problem and gets "fixed" by tuning lr.

WHY THE FITTED QUANTILES LOOK "WRONG" AT THE TAILS, AND ARE NOT. With the
paper's kappa = 1.0 the loss is quadratic within 1.0 of the target, which
pulls the extreme heads toward the median: fitting N(0,1), tau = 0.1 settles
at about -1.00 against a true -1.28, and tau = 0.9 at about +0.99 against
+1.28. That is the Huber smoothing, not a bug -- verified by re-running at
kappa = 0.05, where the same code recovers -1.26 and +1.26, inside 0.02 of
truth. The give-away that the axes are RIGHT is that the two tail errors are
SYMMETRIC. A misaligned tau axis or a transposed u skews them.

HOW NOT TO TEST IT. Do NOT check that the loss is minimised at pred == target
for one fixed target vector. It is not, and should not be: the loss drives each
head to a QUANTILE OF THE TARGET DISTRIBUTION, so against a single N-atom
target the heads converge to that target's quantile spread, not to the target
itself. Testing for identity will make you "fix" correct code. --test-loss
instead fits N quantiles to samples from a known N(5, 3) and checks both the
quantile values and their mean.

WHAT THE DISTRIBUTIONAL TARGET LOOKS LIKE. Per sample the target is itself N
numbers, one per quantile of the NEXT state's distribution:

    T_j = G + gamma^n * mask * theta_j(s', a*)

a* is chosen by the ONLINE net's MEAN over quantiles (Double DQN, stage 03,
unchanged) and the N values are read from the TARGET net at that action. Note
the n-step return G and the per-transition gamma^n are added to every quantile
identically -- a shift of the whole distribution, which is exactly right.

WHAT TO EXPECT: the other big one, alongside stage 04.

DONE WHEN, at 200k on 2 seeds: clearly better return than stage 05, and
--test-loss recovering N(5,3)'s quantiles to within ~0.4 with a mean of ~5.0.

IF IT DOES NOT LEARN, check these, most common first:
  1. acting on raw quantiles       -> argmax must be over .mean(dim=2). An
     argmax over a (B, 17, 51) tensor returns an index into the FLATTENED
     last axes and does not raise.
  2. mean(dim=1) instead of dim=2  -> averages over ACTIONS, giving a
     right-looking shape and an agent that cannot choose.
  3. the three loss traps above.
  4. every earlier stage's failure modes still apply.

Run:
    # the loss unit test, FIRST. Seconds, no env.
    python stage_06_qrdqn/train.py --test-loss

    # local smoke test, about 3 minutes
    python stage_06_qrdqn/train.py --total-steps 2000 \
        --learning-starts 500 --buffer-size 5000

    # the real run
    python stage_06_qrdqn/train.py --total-steps 200000 --seed 1
"""

import argparse
import pathlib
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, bootstrap_mask, make_env   # noqa: E402
from common.metrics import ACHIEVEMENTS, score_run           # noqa: E402
from common.networks import (QRDuelingQNetwork,                # noqa: E402
                             quantile_midpoints)
from common.replay import NStepReplayBuffer, bytes_for       # noqa: E402

STAGE_ID = 'stage_06_qrdqn'


def parse_args(argv=None):
    """Defaults ARE the best hyperparameters, per assignment_v0.pdf 2.1:
    bare `python train.py` must start the run we actually report.

    Identical to stage 05 except for --n-quantiles, so the comparison between
    the two stages has one moving part: the distributional head and its loss.
    """
    p = argparse.ArgumentParser(
        description='Stage 06: QR-DQN (51 quantiles) + n-step + dueling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--logdir', type=str, default='logdir')
    p.add_argument('--device', type=str, default=None,
                   help='cuda / cpu. Default: cuda when available')

    p.add_argument('--test-loss', action='store_true',
                   help='run the quantile-loss unit test and exit. DO THIS '
                        'BEFORE ANY RUN OF THIS STAGE')

    # --- the one new knob -------------------------------------------------
    p.add_argument('--n-quantiles', type=int, default=51,
                   help='quantiles per action. QR-DQN and Rainbow both use 51')
    p.add_argument('--kappa', type=float, default=1.0,
                   help='Huber threshold inside the quantile loss; paper value')
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


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """The QR-DQN loss. Paper eq. 10 / appendix A.

    Args:
      pred:   (B, N) our N quantiles for the action actually taken.
      target: (B, N) the N target quantiles. Treated as samples, no grad.
      taus:   (N,) quantile MIDPOINTS (2i+1)/(2N), from
              common.networks.quantile_midpoints.
      kappa:  Huber threshold. 1.0 in the paper.

    Returns a scalar.

    WHAT IT IS DOING. Ordinary regression pulls a prediction to the MEAN of its
    targets. Quantile regression pulls it to a chosen QUANTILE instead, by
    weighting errors asymmetrically: head tau is penalised tau for being too
    low and (1 - tau) for being too high. At tau = 0.9 being too low hurts 9x
    more, so that head settles at the 90th percentile. Run N heads with
    different taus and you have learned the distribution.

    THE THREE TRAPS, each marked below and each silent. See this module's
    docstring, and verify with --test-loss rather than by reading.
    """
    # Trap 2: THIS ORDER. Every target against every prediction, giving
    # (B, N_target, N_pred). Reversing the two unsqueezes transposes the
    # pairwise matrix and yields a wrong loss that still trains.
    u = target.unsqueeze(-1) - pred.unsqueeze(-2)

    # Huber instead of squared error, so a far-off target cannot produce an
    # unbounded gradient. Quadratic within kappa, linear outside it.
    huber = torch.where(u.abs() <= kappa,
                        0.5 * u.pow(2),
                        kappa * (u.abs() - 0.5 * kappa))

    # The asymmetry IS the method: weight |tau - 1{u < 0}|. u < 0 means the
    # prediction sits ABOVE this target. detach() because the indicator is a
    # weight, not a path for gradient -- leaving it attached does not raise.
    # Trap 1 lives in `taus`: midpoints, never edges.
    weight = (taus.view(1, 1, -1) - (u.detach() < 0).float()).abs()

    elementwise = weight * huber / kappa            # (B, N_target, N_pred)

    # Trap 3: SUM over target quantiles (dim=1), MEAN over prediction
    # quantiles (dim=1 of what is left), then mean over the batch. Swapping
    # the sum and the mean rescales every gradient by N, which reads as a
    # learning-rate problem and gets "fixed" by tuning lr.
    return elementwise.sum(dim=1).mean(dim=1).mean()


def learn(q, q_target, opt, buffer, rng, args, device, taus):
    """One gradient step on the DISTRIBUTIONAL n-step Bellman error.

    Returns (loss, mean Q), where mean Q is still the scalar Q of the chosen
    action -- the mean over its quantiles -- so the number stays comparable
    with every earlier stage's meanQ column.

    WHAT CHANGED FROM STAGE 05, and nothing else did:
      - the nets emit (B, 17, N) instead of (B, 17)
      - the argmax for Double DQN is taken on the MEAN over quantiles
      - the target is N numbers per sample, not one
      - smooth_l1_loss becomes quantile_huber_loss
    """
    obs, act, ret, next_obs, mask, n = buffer.sample(args.batch_size, rng)

    obs = torch.as_tensor(obs, device=device)
    next_obs = torch.as_tensor(next_obs, device=device)
    act = torch.as_tensor(act, device=device)
    ret = torch.as_tensor(ret, device=device)
    mask = torch.as_tensor(mask, device=device)
    n = torch.as_tensor(n, device=device, dtype=torch.float32)

    n_quant = args.n_quantiles

    with torch.no_grad():
        # DOUBLE DQN on the MEAN. q_values() averages over the quantile axis;
        # an argmax on the raw (B, 17, N) tensor does not raise and returns
        # nonsense. Failure mode 1 in the module docstring.
        best = q.q_values(next_obs).argmax(dim=1)                   # (B,)
        # Read all N quantiles of the chosen action from the FROZEN net.
        next_quant = q_target(next_obs)                             # (B,17,N)
        best_idx = best.view(-1, 1, 1).expand(-1, 1, n_quant)
        next_quant = next_quant.gather(1, best_idx).squeeze(1)      # (B, N)

        # The n-step return and the per-transition discount shift the WHOLE
        # distribution: broadcast both over the quantile axis. gamma ** n with
        # the per-transition n, exactly as stage 04 -- unsqueeze(1) so it
        # broadcasts against (B, N) rather than silently forming (B, B).
        target = (ret.unsqueeze(1)
                  + (args.gamma ** n).unsqueeze(1)
                  * mask.unsqueeze(1) * next_quant)                 # (B, N)

    # Our N quantiles for the action we actually took.
    all_quant = q(obs)                                              # (B,17,N)
    act_idx = act.view(-1, 1, 1).expand(-1, 1, n_quant)
    pred = all_quant.gather(1, act_idx).squeeze(1)                  # (B, N)

    loss = quantile_huber_loss(pred, target, taus, kappa=args.kappa)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
    opt.step()
    # mean over quantiles = the Q-value, so this column means what it did in
    # stages 02-05.
    return loss.item(), pred.mean().item()


def test_loss(n_quantiles=8, device='cpu'):
    """Fit N quantiles to samples from a known N(5, 3) and check them.

    THIS IS THE TEST THAT MATTERS FOR THIS STAGE. Run it before any training
    run. Expected, and verified: quantiles recovered to within ~0.4 of
    numpy's, and their mean ~4.98 against a true 5.0.

    Deliberately NOT a check that the loss bottoms out at pred == target --
    see HOW NOT TO TEST IT in the module docstring.
    """
    torch.manual_seed(0)
    taus = quantile_midpoints(n_quantiles, device=device)
    samples = torch.randn(4096, 64, device=device) * 3.0 + 5.0
    pred = torch.zeros(1, n_quantiles, device=device, requires_grad=True)
    opt = torch.optim.Adam([pred], lr=0.05)

    for _ in range(3000):
        batch = samples[torch.randint(0, 4096, (64,), device=device)]
        opt.zero_grad()
        quantile_huber_loss(pred.expand(64, n_quantiles), batch, taus).backward()
        opt.step()

    got = pred[0].detach().cpu().numpy()
    true = np.quantile(samples.cpu().numpy().ravel(), taus.cpu().numpy())
    err = np.abs(got - true).max()

    print(f'fitted quantiles {np.round(got, 2)}')
    print(f'true quantiles   {np.round(true, 2)}')
    print(f'max abs error    {err:.3f}  (expected < 0.5)')
    print(f'mean of fitted   {got.mean():.3f}  (true 5.000)')
    assert err < 0.5, f'quantiles off by {err:.3f}: check taus and the axes'
    assert abs(got.mean() - 5.0) < 0.2, 'the mean is wrong, not just the tails'
    # A monotone fit is the other thing a transposed loss destroys.
    assert np.all(np.diff(got) > 0), 'fitted quantiles are not increasing'
    print('quantile loss OK')


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    # The loss unit test, before anything touches the environment. Run this
    # first on every machine; it is the cheapest way to catch the three traps.
    if args.test_loss:
        test_loss(device=device)
        return

    logdir = (pathlib.Path(args.logdir) / f'crafter_reward-{STAGE_ID}'
              / str(args.seed))
    logdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.seed, logdir)

    # BOTH nets, or load_state_dict below raises. The trunk is identical to
    # stage 05's; only the head's width differs.
    q = QRDuelingQNetwork(N_ACTIONS, n_quantiles=args.n_quantiles).to(device)
    q_target = QRDuelingQNetwork(N_ACTIONS,
                                 n_quantiles=args.n_quantiles).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=args.lr, eps=args.adam_eps)

    # Built once, on the right device. Recomputing it per learn() call would
    # allocate 51 floats a million times for no reason.
    taus = quantile_midpoints(args.n_quantiles, device=device)

    capacity = min(args.buffer_size, args.total_steps)
    # Price it BEFORE allocating: on a box that cannot hold the buffer,
    # allocating it to find out is itself the crash.
    planned = bytes_for(capacity, nstep=True)
    buffer = NStepReplayBuffer(capacity)

    # The window accumulator. maxlen is what makes a full window detectable.
    queue = deque(maxlen=args.n_step)

    print(f'stage {STAGE_ID}  seed {args.seed}  steps {args.total_steps:,}  '
          f'device {device}')
    n_params = sum(p.numel() for p in q.parameters())
    print(f'n_step {args.n_step}  gamma {args.gamma}  '
          f'effective horizon ~{1 / (1 - args.gamma):.0f} steps')
    print(f'quantiles {args.n_quantiles}  kappa {args.kappa}  '
          f'taus[0] {taus[0].item():.4f} (midpoint, not 0)')
    print(f'qr-dueling head: (batch, {N_ACTIONS}, {args.n_quantiles})   '
          f'{n_params:,} parameters')
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
                # q_values() == mean over the quantile axis. An argmax on the
                # raw (1, 17, N) tensor does not raise and is nonsense.
                values = q.q_values(torch.as_tensor(obs[None], device=device))
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
            loss, mean_q = learn(q, q_target, opt, buffer, rng, args,
                                 device, taus)

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
