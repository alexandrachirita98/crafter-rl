"""Stage 07b -- Munchausen alone, on stage 03.

M-DQN, the scalar method of the paper's section 3, not the M-IQN variant stage
07 implements. Same bonus and soft value, acting on one Q-value per action
instead of 51 quantiles:

    target = r + alpha*tau*log pi(a_t|s_t)  +  gamma * mask * sum_a pi(a|s')
             * (Q(s',a) - tau*log pi(a|s'))

The soft bootstrap replaces the Double-DQN argmax, so this row is "Munchausen
vs stage 03", not "Munchausen added to stage 03". _update only.

Run:
    python stage_07b_munchausen_solo/train.py --test-solo
"""

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import (N_ACTIONS, ReplayMemory, get_estimator,
                                main as stage_02_main)
from stage_03_double.train import DoubleDQN
from stage_07_munchausen.train import (ALPHA, CLIP, TAU, munchausen_args,
                                       munchausen_kwargs)

STAGE_ID = 'stage_07b_munchausen_solo'

# Self-test only: scales the fake Q-values so the spread per row is about 2,
# which is where the paper's tau exercises the clip. See test_solo().
Q_SPREAD = 2.0


class MunchausenDoubleDQN(DoubleDQN):
    """Stage 03 with the Munchausen bonus and a soft bootstrap. Scalar M-DQN."""

    def __init__(self, *args, m_alpha=ALPHA, m_tau=TAU, m_clip=CLIP,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._m_alpha, self._m_tau, self._m_clip = m_alpha, m_tau, m_clip

    def _update(self, states, actions, rewards, states_, mask, legal_):
        with torch.no_grad():
            target_qsa = self._target(states, actions, rewards, states_, mask,
                                      legal_)
        qsa = self._estimator(states).gather(1, actions)

        loss = self._td_loss(qsa, target_qsa)
        self._backward(loss)
        self.loss = loss.item()
        self.mean_q = qsa.mean().item()

    def _target(self, states, actions, rewards, states_, mask, legal_=None):
        """r + bonus + gamma * mask * soft value. No argmax anywhere."""
        tgt = self._target_estimator
        alpha, tau, clip = self._m_alpha, self._m_tau, self._m_clip
        # 1. the bonus: alpha * tau * log pi(a_t | s_t), pi from the target
        #    net's Q. The clip matters; unclipped diverges.
        logp = F.log_softmax(tgt(states) / tau, dim=1)
        bonus = (alpha * tau * logp.gather(1, actions)).clamp(min=clip)  # (B,1)
        # 2. the soft bootstrap: expectation under pi of (Q - tau * log pi).
        #    not an argmax -- this replaces stage 03's double target.
        #
        #    legal_ is honoured by renormalising pi over the legal set, which
        #    stage 03 gets for free by masking before an argmax. Without this
        #    a --mask run would have masked acting and UNMASKED bootstrapping,
        #    so it would not be measuring the champion's configuration.
        #    -1e8 before the softmax puts ~0 probability on illegal actions,
        #    and log_softmax renormalises over what is left.
        q_ = tgt(states_)                                               # (B,A)
        if legal_ is not None:
            q_ = q_.masked_fill(legal_ < 0.5, -1e8)
        logp_ = F.log_softmax(q_ / tau, dim=1)
        pi_ = logp_.exp()
        # The masked entries contribute pi*(q - tau*logp) with pi ~ 0 and
        # q = -1e8, i.e. -0.0 rather than NaN, so the sum stays finite. Zero
        # them explicitly so the product is exactly 0 and never 0 * -inf.
        terms = pi_ * (q_ - tau * logp_)
        if legal_ is not None:
            terms = terms * (legal_ >= 0.5)
        soft = terms.sum(1, keepdim=True)                               # (B,1)
        return rewards + bonus + self._gamma * mask * soft


AGENT_CLS = MunchausenDoubleDQN


def extra_args(p):
    munchausen_args(p)
    p.add_argument('--test-solo', action='store_true',
                   help='check the isolation, the clip and the soft value, then exit')


def build(args, capacity):
    """Stage 02's net and one-step buffer. Nothing above stage 03."""
    net = get_estimator(N_ACTIONS, lin_size=args.lin_size, stride=args.stride)
    buffer = ReplayMemory(size=capacity, batch_size=args.batch_size)
    return net, buffer, munchausen_kwargs(args)


# ------------------------------------------------------------- self-test


def test_solo(batch=16, gamma=0.99):
    """Isolation, plus the two checks the cumulative stage makes on quantiles.

    1. rewards = 0, mask = 0: the target is the bonus alone. Every bonus lies
       in [CLIP, 0], and with tau = 0.03 against a Q spread of about 2 some
       rows sit exactly on the clip. If none does, the clip is not wired in.
    2. rewards = 0, mask = 1: (target - bonus) / gamma is the soft value, and
       it is bounded by max_a Q(s',a) + tau*log(17) for every row, because the
       entropy bonus is bounded by log of the action count.
    """
    from stage_04_nstep.train import NStepDoubleDQN
    from stage_06_qrdqn.train import QRDQN

    no_nstep = not issubclass(AGENT_CLS, NStepDoubleDQN)
    no_quant = not issubclass(AGENT_CLS, QRDQN)
    on_double = issubclass(AGENT_CLS, DoubleDQN)
    print(f'inherits from DoubleDQN     {on_double}   want True')
    print(f'agent carries no n-step     {no_nstep}   want True')
    print(f'agent carries no quantiles  {no_quant}   want True')

    torch.manual_seed(0)
    agent = AGENT_CLS.__new__(AGENT_CLS)
    agent._gamma = gamma
    agent._m_alpha, agent._m_tau, agent._m_clip = ALPHA, TAU, CLIP
    fixed = torch.randn(batch, N_ACTIONS) * Q_SPREAD
    agent._target_estimator = lambda x: fixed

    states = torch.zeros(batch, 3, 64, 64, dtype=torch.uint8)
    actions = torch.randint(0, N_ACTIONS, (batch, 1))
    zeros, ones = torch.zeros(batch, 1), torch.ones(batch, 1)

    t0 = agent._target(states, actions, zeros, states, zeros)
    in_range = bool(((t0 >= CLIP - 1e-6) & (t0 <= 1e-6)).all())
    on_clip = int((t0 <= CLIP + 1e-6).sum())
    print(f'\nmask=0 -> target is the bonus, within [{CLIP}, 0]  '
          f'{in_range}   want True')
    print(f'rows sitting on the clip              {on_clip}/{batch}  want > 0')

    t1 = agent._target(states, actions, zeros, states, ones)
    soft = (t1 - t0) / gamma
    max_q = fixed.max(1, keepdim=True)[0]
    bound = max_q + TAU * torch.log(torch.tensor(float(N_ACTIONS)))
    within = bool((soft <= bound + 1e-5).all())
    print(f'\nsoft value  {soft.mean():.3f}   max-Q {max_q.mean():.3f}   '
          f'bound {bound.mean():.3f}')
    print(f'soft <= max_a Q + tau*log(17)         {within}   want True')

    above_mean = bool((soft >= fixed.mean(1, keepdim=True) - 1e-5).all())
    print(f'soft >= mean_a Q                      {above_mean}   want True')

    # At tau = 0.03 the softmax is numerically a hard argmax once the Q spread
    # exceeds ~0.3, so soft == max_q above and that is the paper's intent --
    # the entropy term is a small regulariser, not a smoothed bootstrap.
    # Testing "soft < max_q" at the default spread would fail on correct code.
    # The expectation only shows itself when the Q values are close, so check
    # it in the regime where it is actually distinguishable from an argmax.
    tight = torch.randn(batch, N_ACTIONS) * (TAU / 3.0)
    agent._target_estimator = lambda x: tight
    t_tight = agent._target(states, actions, zeros, states, ones)
    b_tight = agent._target(states, actions, zeros, states, zeros)
    soft_t = (t_tight - b_tight) / gamma
    strictly_above = bool((soft_t > tight.max(1, keepdim=True)[0] + 1e-6).all())
    print(f'\nwith Q spread {TAU / 3.0:.3f} (<< tau), soft value sits strictly')
    print(f'above max_a Q (the entropy term)      {strictly_above}   want True')

    # The masked bootstrap. pi is renormalised over the legal set, so the
    # soft value must ignore an illegal action even when it holds the largest
    # Q. Make the argmax illegal and check the value drops to the best legal
    # one; without the masked_fill this returns the illegal action's value and
    # a --mask run would bootstrap off actions it can never take.
    agent._target_estimator = lambda x: fixed
    best = fixed.argmax(1, keepdim=True)
    legal = torch.ones(batch, N_ACTIONS)
    legal.scatter_(1, best, 0.0)                    # forbid the argmax
    t_masked = agent._target(states, actions, zeros, states, ones, legal)
    b_masked = agent._target(states, actions, zeros, states, zeros, legal)
    soft_m = (t_masked - b_masked) / gamma
    legal_max = fixed.masked_fill(legal < 0.5, float('-inf')).max(1,
                                                                  keepdim=True)[0]
    finite = bool(torch.isfinite(soft_m).all())
    ignores_illegal = bool((soft_m <= max_q - 1e-3).all())
    tracks_legal = bool((soft_m - legal_max).abs().max() < 0.05)
    print(f'\nwith the argmax masked out, soft value is finite  '
          f'{finite}   want True')
    print(f'  below the illegal max ({max_q.mean():.3f})            '
          f'{ignores_illegal}   want True')
    print(f'  tracks the best LEGAL Q ({legal_max.mean():.3f})         '
          f'{tracks_legal}   want True')

    ok = (on_double and no_nstep and no_quant and in_range and on_clip > 0
          and within and above_mean and strictly_above
          and finite and ignores_illegal and tracks_legal)
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-solo' in (argv if argv is not None else sys.argv[1:]):
        return test_solo()
    return stage_02_main(argv, agent_cls=AGENT_CLS, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
