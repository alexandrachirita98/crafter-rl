"""Stage 07 -- Munchausen RL on top of QR-DQN.

Vieillard, Pietquin & Geist 2020, https://arxiv.org/abs/2007.14430, section 3
plus the hyperparameters.

Add to the reward a bonus for having done what you already believed was good:
alpha * tau * log pi(a|s), with pi the softmax of your own Q-values. It acts
as an implicit KL regulariser that stops the policy thrashing between updates.

_target_quantiles only, but note two things. The soft bootstrap replaces the
Double-DQN argmax -- the next-state value is an expectation under pi, not the
value of one action -- so from here on "double" is gone. And with quantiles
the bonus is a scalar per sample broadcast over N, with the soft value taken
per quantile and pi from the mean: the M-IQN construction from the appendix.

With n-step, the bonus is for the first action of the window.

Run:
    python stage_07_munchausen/train.py --test-munchausen
    python stage_07_munchausen/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import main as stage_02_main
from stage_06_qrdqn.train import (QRDQN, build as build_06,
                                  extra_args as extra_06)

STAGE_ID = 'stage_07_munchausen'

ALPHA, TAU, CLIP = 0.9, 0.03, -1.0      # paper defaults, Atari scale.

# Why these are tunable now. The defaults are the paper's Atari constants.
# Fed realistic Crafter Q-spreads the bonus is -0.11 to -0.84 per step and
# sits on the -1.0 clip 34-72% of the time once the spread reaches 0.5, while
# Crafter's real reward is about +0.013 per step -- the bonus is 8-65x the
# signal with the opposite sign, which no amount of budget fixes. --m-tau 0.3
# or --m-clip -0.05 (the per-step reward scale) are the arms worth screening.

# Self-test only: scales the fake quantiles so the mean-Q spread per row is
# about 2, which is where the paper's tau exercises the clip. See
# test_munchausen().
Q_SPREAD = 8.0


class MunchausenQRDQN(QRDQN):

    def __init__(self, *args, m_alpha=ALPHA, m_tau=TAU, m_clip=CLIP, **kwargs):
        super().__init__(*args, **kwargs)
        self._m_alpha, self._m_tau, self._m_clip = m_alpha, m_tau, m_clip

    def _target_quantiles(self, states, actions, rewards, states_, mask, n):
        tgt = self._target_estimator
        alpha, tau, clip = self._m_alpha, self._m_tau, self._m_clip
        # 1. the bonus: alpha * tau * log pi(a_t | s_t), pi from the target
        #    net's mean Q. The clip matters; unclipped diverges.
        logp = F.log_softmax(tgt(states).mean(2) / tau, dim=1)
        bonus = (alpha * tau * logp.gather(1, actions)).clamp(min=clip)  # (B,1)
        # 2. the soft bootstrap: expectation under pi of
        #    (quantiles - tau * log pi), not an argmax.
        quant_ = tgt(states_)                                           # (B,A,N)
        logp_ = F.log_softmax(quant_.mean(2) / tau, dim=1)              # (B,A)
        pi_ = logp_.exp()
        soft = (pi_.unsqueeze(2)
                * (quant_ - tau * logp_.unsqueeze(2))).sum(1)           # (B,N)
        return rewards + bonus + (self._gamma ** n) * mask * soft


def munchausen_args(p):
    """The three constants, shared with stage 07b."""
    p.add_argument('--m-alpha', type=float, default=ALPHA,
                   help='Munchausen bonus scale')
    p.add_argument('--m-tau', type=float, default=TAU,
                   help='entropy temperature; 0.03 is the Atari default, '
                        '0.3 is the Crafter arm worth screening')
    p.add_argument('--m-clip', type=float, default=CLIP,
                   help='floor on the bonus; -0.05 is the per-step reward scale')


def munchausen_kwargs(args):
    """-> the dict build() merges into the agent's constructor kwargs."""
    return {'m_alpha': args.m_alpha, 'm_tau': args.m_tau,
            'm_clip': args.m_clip}


def extra_args(p):
    extra_06(p)
    munchausen_args(p)
    p.add_argument('--test-munchausen', action='store_true',
                   help='check the bonus clip and the soft value, then exit')


def build(args, capacity):
    """Stage 06's build, plus the three Munchausen constants as kwargs."""
    net, buffer, extra = build_06(args, capacity)
    return net, buffer, dict(extra, **munchausen_kwargs(args))


# ------------------------------------------------------------- self-test


def _fake_agent(n_quant, gamma, fixed, alpha=ALPHA, tau=TAU, clip=CLIP):
    """The agent's target maths with no network and no buffer."""
    agent = MunchausenQRDQN.__new__(MunchausenQRDQN)
    agent._gamma, agent._n_quant = gamma, n_quant
    agent._m_alpha, agent._m_tau, agent._m_clip = alpha, tau, clip
    agent._target_estimator = lambda x: fixed
    return agent


def test_munchausen(batch=16, n_quant=51, gamma=0.99):
    """Build the agent without a network and check the two target pieces.

    1. rewards = 0, mask = 0: the target is the bonus alone, repeated over
       every quantile. Every bonus lies in [CLIP, 0] and some rows should sit
       exactly on the clip; if none does, the clip is not wired in. The spread
       has to be built deliberately -- averaging 51 standard-normal quantiles
       leaves a per-action spread near 0.5, where the raw bonus bottoms out
       near -0.6, so Q_SPREAD scales it into the paper's tuned range.
    2. rewards = 0, mask = 1, n = 1: (target - bonus) / gamma is the soft
       value, whose mean over quantiles is at most max_a Q(s',a) + tau*log(17),
       the entropy bonus being bounded by log of the action count.
    3. --m-clip reaches the maths: a shallower floor bounds the bonus above
       the paper's, which is the whole point of exposing the flag.
    """
    torch.manual_seed(0)
    fixed = torch.randn(batch, 17, n_quant) * Q_SPREAD
    agent = _fake_agent(n_quant, gamma, fixed)

    states = torch.zeros(batch, 3, 64, 64, dtype=torch.uint8)
    actions = torch.randint(0, 17, (batch, 1))
    zeros = torch.zeros(batch, 1)
    ones = torch.ones(batch, 1)

    # 1. the bonus alone
    t0 = agent._target_quantiles(states, actions, zeros, states, zeros, ones)
    bonus = t0[:, :1]
    flat = bool((t0 - bonus).abs().max() < 1e-6)
    in_range = bool(((bonus >= CLIP - 1e-6) & (bonus <= 0 + 1e-6)).all())
    on_clip = int((bonus <= CLIP + 1e-6).sum())
    print(f'bonus broadcast over {n_quant} quantiles  {flat}   want True')
    print(f'bonus within [{CLIP}, 0]                 {in_range}   want True')
    print(f'rows sitting on the clip              {on_clip}/{batch}  want > 0')

    # 2. the soft value against the entropy bound
    t1 = agent._target_quantiles(states, actions, zeros, states, ones, ones)
    soft = ((t1 - bonus) / gamma).mean(1)
    max_q = fixed.mean(2).max(1)[0]
    bound = max_q + TAU * torch.log(torch.tensor(17.0))
    within = bool((soft <= bound + 1e-5).all())
    print(f'\nsoft value  {soft.mean():.3f}   max-Q {max_q.mean():.3f}   '
          f'bound {bound.mean():.3f}')
    print(f'soft <= max_a Q + tau*log(17) for every row  {within}   want True')

    # 3. the exposed constants are not decoration
    shallow = _fake_agent(n_quant, gamma, fixed, clip=-0.05)
    b2 = shallow._target_quantiles(states, actions, zeros, states, zeros,
                                   ones)[:, :1]
    clip_wired = bool((b2 >= -0.05 - 1e-6).all() and b2.min() > bonus.min())
    print(f'\n--m-clip -0.05 raises the floor  {clip_wired}   want True')
    print(f'    bonus min: {bonus.min():.3f} at clip {CLIP}, '
          f'{b2.min():.3f} at clip -0.05')

    ok = flat and in_range and on_clip > 0 and within and clip_wired
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-munchausen' in (argv if argv is not None else sys.argv[1:]):
        return test_munchausen()
    return stage_02_main(argv, agent_cls=MunchausenQRDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
