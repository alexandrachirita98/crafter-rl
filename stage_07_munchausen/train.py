"""Stage 07 -- Munchausen DQN.

WHERE: 💻 `--test-munchausen` then a 2000-step smoke test, then RunPod for
200k x 2 seeds (about 30 min, about $0.18). Never launch a pod to find out
whether it runs.

WHAT: stage 06 with the TARGET changed and nothing else. The trunk, the
quantile head, the quantile Huber loss, the n-step accumulator, the buffer and
every hyperparameter are stage 06 verbatim. Three new constants arrive
(alpha, tau, clip) and they are the paper's defaults.

    Vieillard, Pietquin & Geist 2020, "Munchausen Reinforcement Learning",
    https://arxiv.org/abs/2007.14430, section 3 plus the hyperparameters

THE IDEA, WHICH SOUNDS CIRCULAR AND IS NOT. Add to the reward a bonus for
having done what you already believed was good:

    r' = r + alpha * tau * log pi(a|s)

where pi is the softmax of your own Q-values at temperature tau. The agent
rewards itself for acting consistently with its own current policy -- hence
"Munchausen", pulling yourself up by your own hair.

WHY IT WORKS. The bonus is a log-policy term, so adding it is equivalent to
penalising the KL divergence between consecutive policies: it is an implicit
KL regulariser. Without it, each target-network refresh can swing the greedy
policy to a completely different action and the agent thrashes between
updates, relearning from scratch. With it, moving away from what you currently
believe costs something, so the policy changes smoothly. The paper's claim --
and the reason it is in this stack -- is that this single line added to DQN
beats several full algorithms.

THE TWO HALVES, AND THE SECOND IS THE ONE PEOPLE SKIP.

  1. THE BONUS. alpha * tau * log pi(a_t | s_t), using the action ACTUALLY
     TAKEN, with log pi from the TARGET net. Clipped below at -1.

  2. THE SOFT BOOTSTRAP. The next-state value is NOT a max. It is the
     entropy-regularised expectation under pi:

         soft_v(s') = sum_a pi(a|s') * (Q(s',a) - tau * log pi(a|s'))

     This is part of the method, not an optional refinement: the bonus is
     derived from a maximum-entropy formulation, and bootstrapping with a hard
     max against an entropy-regularised bonus mixes two different objectives.
     Keeping the max is the single most common way to implement M-DQN and get
     a null result.

WHY THE CLIP MATTERS. log pi is unbounded below: an action the policy has
essentially ruled out has log pi -> -infinity, and at tau = 0.03 the raw bonus
can be enormous and negative. Early in training epsilon-greedy takes exactly
such actions constantly, so without the clip the targets blow up and training
diverges. clamp(min=-1) bounds it. The paper does this; do not remove it.

THE DISTRIBUTIONAL DETAIL THE GUIDE'S SNIPPET DOES NOT COVER, AND THE ONE
THING TO GET RIGHT HERE. The guide gives Munchausen for a SCALAR-valued net.
Our stage 06 net emits 51 quantiles per action, so the target is (B, N), not
(B,). The resolution:

  - The POLICY is defined on scalar Q-values. pi = softmax(Q/tau) where
    Q = mean over quantiles. A softmax over a (B, 17, 51) tensor is
    meaningless -- there is no distribution over quantiles to take.
  - The BONUS is a scalar per sample. It shifts the whole target distribution,
    so it broadcasts over the quantile axis -- exactly like the n-step return
    G already does in stage 06.
  - The SOFT BOOTSTRAP is taken per quantile, with the pi weights computed
    from the scalar Q's: for quantile j,

        soft_v_j(s') = sum_a pi(a|s') * (theta_j(s',a) - tau * log pi(a|s'))

    i.e. the SAME pi mixes the 51-vectors of each action. That keeps the
    distributional information (each action contributes its own spread) while
    the policy stays a proper distribution over the 17 actions.

NOTE WHAT DISAPPEARS: Double DQN's argmax. M-DQN's bootstrap is an
expectation under pi over ALL actions, so there is no action selection step
left to decouple -- the overestimation Double DQN fixed is handled by the soft
expectation instead. That is a real change from stage 06 and worth a sentence
in the deck; it is not an accidental removal.

HYPERPARAMETERS: alpha = 0.9, tau = 0.03, clip = -1. The paper's values.
DO NOT TUNE THEM before everything else works -- and if a sweep happens at
all, it is alpha, after 200k, one value at a time.

WHAT TO EXPECT: medium-big, per the guide's stage table.

DONE WHEN, at 200k on 2 seeds: better return than stage 06, and
--test-munchausen passing. If return is flat but stable, report it as a null
and keep the stage on disk as evidence -- contribution.md section 6 wants the
reverts in the slides.

IF IT DOES NOT LEARN, check these, most common first:
  1. a hard max in the bootstrap instead of the soft expectation -> the
     single most common M-DQN bug. Runs fine, learns like stage 06.
  2. the clip missing              -> loss and meanQ blow up within the first
     few thousand gradient steps. The clearest failure to spot.
  3. log pi taken from the ONLINE net -> both log-policy terms come from the
     TARGET net in the paper; using the online net puts the policy inside its
     own gradient and is unstable.
  4. the bonus using argmax Q instead of the action actually taken -> the
     bonus is about what the agent DID, not what it would do now.
  5. softmax over the quantile axis -> see the distributional note above.
  6. every earlier stage's failure modes still apply.

Run:
    # the target unit test, FIRST. Seconds, no env.
    python stage_07_munchausen/train.py --test-munchausen

    # the stage 06 loss test still has to pass -- the loss is unchanged
    python stage_07_munchausen/train.py --test-loss

    # local smoke test, about 3 minutes
    python stage_07_munchausen/train.py --total-steps 2000 \
        --learning-starts 500 --buffer-size 5000

    # the real run
    python stage_07_munchausen/train.py --total-steps 200000 --seed 1
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
from common.networks import (QRDuelingQNetwork,                # noqa: E402
                             quantile_midpoints)
from common.replay import NStepReplayBuffer, bytes_for       # noqa: E402

STAGE_ID = 'stage_07_munchausen'


def parse_args(argv=None):
    """Defaults ARE the best hyperparameters, per assignment_v0.pdf 2.1:
    bare `python train.py` must start the run we actually report.

    Identical to stage 06 except for the three Munchausen constants, so the
    comparison between the two stages has one moving part: the target.

    The three are exposed as flags so they appear in --help and in the run
    log, NOT because they should be swept. Paper defaults; see the module
    docstring.
    """
    p = argparse.ArgumentParser(
        description='Stage 07: Munchausen QR-DQN on Crafter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='environment steps; the benchmark budget is 1M')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--logdir', type=str, default='logdir')
    p.add_argument('--device', type=str, default=None,
                   help='cuda / cpu. Default: cuda when available')

    p.add_argument('--test-loss', action='store_true',
                   help='run the quantile-loss unit test and exit')
    p.add_argument('--test-munchausen', action='store_true',
                   help='run the Munchausen-target unit test and exit. DO '
                        'THIS BEFORE ANY RUN OF THIS STAGE')

    # --- Munchausen: the three new constants, all paper defaults ----------
    p.add_argument('--m-alpha', type=float, default=0.9,
                   help='weight on the log-policy bonus. Paper: 0.9')
    p.add_argument('--m-tau', type=float, default=0.03,
                   help='entropy temperature for pi = softmax(Q/tau). '
                        'Paper: 0.03. NOT the quantile taus')
    p.add_argument('--m-clip', type=float, default=-1.0,
                   help='lower clamp on the bonus. Paper: -1. REMOVING THIS '
                        'DIVERGES')

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

    THE THREE TRAPS, each marked below and each silent. Verify with
    --test-loss rather than by reading. Unchanged from stage 06, including
    the tail behaviour at kappa = 1.0: fitting N(0,1), tau = 0.1 settles near
    -1.00 against a true -1.28 because the Huber region pulls the extreme
    heads toward the median. At kappa = 0.05 the same code gives -1.26.
    SYMMETRIC tail errors mean the axes are right; skewed ones do not.
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


def munchausen_target(next_quant, cur_q, next_q, act, ret, mask, gamma_n,
                      args):
    """The Munchausen n-step distributional target. (B, N).

    Args:
      next_quant: (B, 17, N) TARGET net's quantiles at s_{t+n}.
      cur_q:      (B, 17)    TARGET net's mean Q at s_t. For the bonus.
      next_q:     (B, 17)    TARGET net's mean Q at s_{t+n}. For pi and soft_v.
      act:        (B,)       the action ACTUALLY TAKEN at s_t.
      ret:        (B,)       the n-step return G_t.
      mask:       (B,)       0.0 only on actual death. From common.env.
      gamma_n:    (B,)       gamma ** n, the PER-TRANSITION discount.
      args:       for m_alpha, m_tau, m_clip.

    Everything here is under no_grad at the call site. Both log-policies come
    from the TARGET net, per the paper.

    THE SHAPE RULE THAT KEEPS THIS HONEST: the policy is a distribution over
    the 17 ACTIONS, so every softmax/log_softmax below is dim=1 on a (B, 17)
    tensor of MEAN Q-values. The quantile axis never enters a softmax.
    """
    alpha, tau = args.m_alpha, args.m_tau

    # --- half 1: the bonus, a scalar per sample --------------------------
    # log pi(.|s_t) at the CURRENT state, from the target net.
    logp_cur = F.log_softmax(cur_q / tau, dim=1)                   # (B, 17)
    # The action actually taken -- not the argmax. The bonus is about what the
    # agent DID.
    bonus = alpha * tau * logp_cur.gather(1, act.unsqueeze(1)).squeeze(1)
    # THE CLIP. log pi is unbounded below, and epsilon-greedy picks ruled-out
    # actions constantly early on. Without this the target explodes.
    bonus = bonus.clamp(min=args.m_clip)                           # (B,)

    # --- half 2: the soft bootstrap, per quantile ------------------------
    logp_next = F.log_softmax(next_q / tau, dim=1)                 # (B, 17)
    pi_next = logp_next.exp()                                      # (B, 17)

    # soft_v_j = sum_a pi(a|s') * (theta_j(s',a) - tau * log pi(a|s'))
    # NOT a max over actions. The entropy term -tau*log pi is what makes this
    # the entropy-regularised value the bonus is derived from; a hard max here
    # is the most common way to implement M-DQN and measure nothing.
    # pi and log pi are per-action scalars, so they broadcast over the
    # quantile axis: unsqueeze(2) against (B, 17, N).
    soft = next_quant - tau * logp_next.unsqueeze(2)               # (B, 17, N)
    soft_v = (pi_next.unsqueeze(2) * soft).sum(dim=1)              # (B, N)

    # --- assemble ---------------------------------------------------------
    # ret and bonus are scalars per sample: they shift the whole distribution,
    # so they broadcast over the quantile axis. unsqueeze(1) on every (B,)
    # term, or (B,) against (B, N) silently broadcasts into (B, B)-shaped
    # nonsense in the general case.
    return (ret.unsqueeze(1) + bonus.unsqueeze(1)
            + gamma_n.unsqueeze(1) * mask.unsqueeze(1) * soft_v)   # (B, N)


def learn(q, q_target, opt, buffer, rng, args, device, taus):
    """One gradient step on the Munchausen distributional n-step error.

    Returns (loss, mean Q). The PREDICTION side is stage 06 verbatim -- same
    net, same gather, same quantile Huber loss. Only the target changed.
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
        # TWO forward passes of the frozen net, where stage 06 needed one.
        # cur_q is the new one: the bonus is log pi(a_t | s_t), so Munchausen
        # genuinely has to evaluate the target net at s_t as well as s_{t+n}.
        # So a gradient step costs three big forwards here (two frozen, one
        # online) against stage 06's two. Expect somewhat fewer fps than
        # stage 06 at the same settings; that is the method, not a bug.
        next_quant = q_target(next_obs)                      # (B, 17, N)
        next_q = next_quant.mean(dim=2)                      # (B, 17)
        cur_q = q_target(obs).mean(dim=2)                    # (B, 17)

        # No argmax anywhere: M-DQN bootstraps from an expectation under pi
        # over all 17 actions, so Double DQN's select/evaluate split has
        # nothing left to decouple. See the module docstring.
        target = munchausen_target(next_quant, cur_q, next_q, act, ret, mask,
                                   args.gamma ** n, args)         # (B, N)

    all_quant = q(obs)                                            # (B, 17, N)
    act_idx = act.view(-1, 1, 1).expand(-1, 1, n_quant)
    pred = all_quant.gather(1, act_idx).squeeze(1)                # (B, N)

    loss = quantile_huber_loss(pred, target, taus, kappa=args.kappa)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
    opt.step()
    return loss.item(), pred.mean().item()


def test_munchausen(device='cpu'):
    """Check the four properties of the target that silent bugs break.

    Fast, no environment, no network. Run before any training run.
    """
    torch.manual_seed(0)
    B, A, N = 8, N_ACTIONS, 11

    class _Args:
        m_alpha, m_tau, m_clip = 0.9, 0.03, -1.0

    args = _Args()
    next_quant = torch.randn(B, A, N, device=device)
    next_q = next_quant.mean(dim=2)
    cur_q = torch.randn(B, A, device=device)
    act = torch.randint(0, A, (B,), device=device)
    ret = torch.rand(B, device=device)
    mask = torch.ones(B, device=device)
    gamma_n = torch.full((B,), 0.99 ** 10, device=device)

    t = munchausen_target(next_quant, cur_q, next_q, act, ret, mask,
                          gamma_n, args)

    # 1. SHAPE. (B, N) -- one target per quantile, not per action.
    assert t.shape == (B, N), f'target shape {tuple(t.shape)}, expected {(B, N)}'

    # 2. THE CLIP BINDS. Build a cur_q where the taken action is hopeless:
    # at tau = 0.03 a gap of 1.0 in Q is a log-prob of about -33, so the raw
    # bonus would be about -0.9 * 33 = -30. Clipped, it must be exactly -1.
    cur_bad = torch.zeros(B, A, device=device)
    cur_bad[torch.arange(B), act] = -1.0            # taken action is worst
    other = (act + 1) % A
    cur_bad[torch.arange(B), other] = 1.0
    t_clip = munchausen_target(next_quant, cur_bad, next_q, act, ret,
                              torch.zeros(B, device=device), gamma_n, args)
    # mask = 0 kills the bootstrap, so the target is exactly ret + bonus.
    implied_bonus = t_clip[:, 0] - ret
    assert torch.allclose(implied_bonus,
                          torch.full((B,), args.m_clip, device=device),
                          atol=1e-5), \
        f'clip not binding: bonus {implied_bonus[:3].tolist()}, want -1.0'

    # 3. IT IS A SOFT EXPECTATION, NOT A MAX. With a near-uniform policy
    # (large tau) the bootstrap must sit near the MEAN over actions, well
    # below the max. A hard-max implementation fails this.
    class _Hot:
        m_alpha, m_tau, m_clip = 0.0, 100.0, -1.0    # alpha 0: bonus off
    flat = munchausen_target(next_quant, cur_q, next_q, act,
                             torch.zeros(B, device=device),
                             torch.ones(B, device=device),
                             torch.ones(B, device=device), _Hot())
    # Compare the per-quantile bootstrap against the per-quantile mean and max
    # over actions. The entropy term shifts it up by tau*H, so compare the
    # ORDERING rather than exact values.
    mean_over_actions = next_quant.mean(dim=1)                 # (B, N)
    max_over_actions = next_quant.max(dim=1).values            # (B, N)
    # soft_v with a near-uniform pi tracks the mean, not the max.
    d_mean = (flat - flat.mean() - (mean_over_actions - mean_over_actions.mean())).abs().mean()
    d_max = (flat - flat.mean() - (max_over_actions - max_over_actions.mean())).abs().mean()
    assert d_mean < d_max, \
        f'bootstrap tracks the max ({d_max:.3f}) not the mean ({d_mean:.3f}): hard max?'

    # 4. DEATH KILLS THE BOOTSTRAP BUT NOT THE BONUS. With mask = 0 the
    # target must not depend on s_{t+n} at all.
    t_dead_a = munchausen_target(next_quant, cur_q, next_q, act, ret,
                                 torch.zeros(B, device=device), gamma_n, args)
    t_dead_b = munchausen_target(torch.randn_like(next_quant), cur_q,
                                 torch.randn_like(next_q), act, ret,
                                 torch.zeros(B, device=device), gamma_n, args)
    assert torch.allclose(t_dead_a, t_dead_b, atol=1e-6), \
        'mask=0 target still depends on the next state'

    print(f'target shape      {tuple(t.shape)} (B, N)')
    print(f'clip binds        implied bonus {implied_bonus[0].item():.4f} '
          f'(expected {args.m_clip})')
    print(f'soft not max      |d-mean| {d_mean:.3f} < |d-max| {d_max:.3f}')
    print(f'mask=0 isolates   next state ignored: OK')
    print(f'alpha {args.m_alpha}  tau {args.m_tau}  clip {args.m_clip}')
    print('munchausen target OK')


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
    if args.test_munchausen:
        test_munchausen(device=device)
        return

    logdir = (pathlib.Path(args.logdir) / f'crafter_reward-{STAGE_ID}'
              / str(args.seed))
    logdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.seed, logdir)

    # Identical to stage 06: same trunk, same quantile head, same width.
    # Munchausen changes the target, not the architecture.
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
    print(f'munchausen: alpha {args.m_alpha}  tau {args.m_tau}  '
          f'clip {args.m_clip}  (soft bootstrap, no argmax)')
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
