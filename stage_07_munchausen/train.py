"""Stage 07 -- Munchausen RL on top of QR-DQN.

Vieillard, Pietquin & Geist 2020, https://arxiv.org/abs/2007.14430, section 3
plus the hyperparameters.

Add to the reward a bonus for having done what you already believed was good:
alpha * tau * log pi(a|s), with pi the softmax of your own Q-values. It sounds
circular and it works, acting as an implicit KL regulariser that stops the
policy thrashing between updates.

_target_quantiles only. Two things to notice:

  - The soft bootstrap REPLACES the Double-DQN argmax. The next-state value is
    an expectation under pi, not the value of one chosen action. That is part
    of the method, not an optional refinement, so from here on "double" is
    gone and the deck's stage table should say so.
  - With quantiles the bonus is a scalar per sample broadcast over the N
    quantiles, and the soft value is taken per quantile with pi from the mean.
    This is the M-IQN construction from the paper's appendix.

With n-step, the bonus is for the FIRST action of the window, which is the
action the transition is about.

Run:
    python stage_07_munchausen/train.py --test-munchausen
    python stage_07_munchausen/train.py --total-steps 2000 --warmup-steps 500 \
        --buffer-size 5000
    python stage_07_munchausen/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from stage_02_dqn.train import main as stage_02_main
from stage_06_qrdqn.train import QRDQN, build, extra_args as extra_06

STAGE_ID = 'stage_07_munchausen'

ALPHA, TAU, CLIP = 0.9, 0.03, -1.0      # paper defaults. Do not tune first.

# Self-test only: scales the fake quantiles so the mean-Q spread per row is
# about 2, which is where the paper's tau exercises the clip. See
# test_munchausen().
Q_SPREAD = 8.0


class MunchausenQRDQN(QRDQN):

    def _target_quantiles(self, states, actions, rewards, states_, mask, n):
        tgt = self._target_estimator
        # 1. the bonus: alpha * tau * log pi(a_t | s_t), pi from the target
        #    net's mean Q. The clip matters; unclipped diverges.
        logp = F.log_softmax(tgt(states).mean(2) / TAU, dim=1)
        bonus = (ALPHA * TAU * logp.gather(1, actions)).clamp(min=CLIP)  # (B,1)
        # 2. the soft bootstrap: expectation under pi of
        #    (quantiles - tau * log pi). NOT an argmax.
        quant_ = tgt(states_)                                           # (B,A,N)
        logp_ = F.log_softmax(quant_.mean(2) / TAU, dim=1)              # (B,A)
        pi_ = logp_.exp()
        soft = (pi_.unsqueeze(2)
                * (quant_ - TAU * logp_.unsqueeze(2))).sum(1)           # (B,N)
        return rewards + bonus + (self._gamma ** n) * mask * soft


def extra_args(p):
    extra_06(p)
    p.add_argument('--test-munchausen', action='store_true',
                   help='check the bonus clip and the soft value, then exit')


# ------------------------------------------------------------- self-test


def test_munchausen(batch=16, n_quant=51, gamma=0.99):
    """Build the agent without a network and check the two target pieces.

    1. rewards = 0, mask = 0: the target is the bonus alone, repeated over
       every quantile. Every bonus lies in [CLIP, 0], and with tau = 0.03
       against a mean-Q spread of about 2 some rows sit exactly on the clip.
       If none does, the clip is not wired in.

       The spread has to be built deliberately: averaging 51 standard-normal
       quantiles leaves a per-action spread near 0.5, where the raw bonus
       bottoms out around -0.6 and never reaches -1. Scaling the fixed
       quantiles by Q_SPREAD puts it in the range the paper's defaults are
       tuned for.
    2. rewards = 0, mask = 1, n = 1: (target - bonus) / gamma is the soft
       value. Its mean over quantiles is at most max_a Q(s',a) + tau*log(17)
       for every row, because the entropy bonus is bounded by log of the
       action count.
    """
    torch.manual_seed(0)
    agent = MunchausenQRDQN.__new__(MunchausenQRDQN)
    agent._gamma, agent._n_quant = gamma, n_quant
    fixed = torch.randn(batch, 17, n_quant) * Q_SPREAD
    agent._target_estimator = lambda x: fixed

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

    ok = flat and in_range and on_clip > 0 and within
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-munchausen' in (argv if argv is not None else sys.argv[1:]):
        return test_munchausen()
    return stage_02_main(argv, agent_cls=MunchausenQRDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
