"""Stage 06 -- QR-DQN: 51 quantiles per action instead of one mean.

Dabney, Rowland, Bellemare & Munos 2018, https://arxiv.org/abs/1710.10044,
section 3 and Appendix A.

Output N numbers describing the spread of outcomes rather than one mean. The
agent still acts on the mean, but the network has N times more to learn from
per sample, which is why it helps on a small budget. Unlike C51 it needs no
[V_min, V_max] guess and no projection step.

Output becomes (B, 17, N); step acts on the mean over N; _update regresses N
quantiles with the quantile Huber loss. The target is its own method because
stage 07 replaces it, the per-sample loss because stages 08 and 10 reuse it.

Run:
    python stage_06_qrdqn/train.py --test-loss
    python stage_06_qrdqn/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.masking import random_legal
from common.networks import quantile_midpoints     # (2i+1)/(2N), tested
from stage_02_dqn.train import (DEVICE, N_ACTIONS, _apply_mask,
                                _mask_or_ones, _to_tensor,
                                main as stage_02_main)
from stage_04_nstep.train import NStepDoubleDQN, build as build_04
from stage_05_dueling.train import DuelingNet, extra_args as extra_05

STAGE_ID = 'stage_06_qrdqn'


class QRDuelingNet(DuelingNet):
    """Stage 05's net with every output N quantiles wide: (B, A, N)."""

    def __init__(self, action_num, n_quantiles=51, lin_size=64, stride=1):
        super().__init__(action_num, lin_size=lin_size, stride=stride)
        self.action_num, self.n_quantiles = action_num, n_quantiles
        self.value[-1] = nn.Linear(lin_size, n_quantiles)
        self.adv[-1] = nn.Linear(lin_size, action_num * n_quantiles)

    def forward(self, x):
        f = self.trunk(x)
        v = self.value(f).view(-1, 1, self.n_quantiles)
        a = self.adv(f).view(-1, self.action_num, self.n_quantiles)
        # dim=1 is the action axis. Using dim=2 zeroes each action's mean
        # return -- exactly what the agent acts on -- and it does not crash.
        return v + a - a.mean(1, keepdim=True)


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """pred (B,N) our quantiles; target (B,N'); taus (N,). -> (B,) per sample."""
    # every target against every prediction -> (B, N_target, N_pred)
    u = target.unsqueeze(2) - pred.unsqueeze(1)
    huber = torch.where(u.abs() <= kappa,
                        0.5 * u.pow(2),
                        kappa * (u.abs() - 0.5 * kappa))
    # The asymmetry is the point: quantile tau is weighted |tau - 1{u<0}|, so
    # a low-tau head learns to sit low and a high-tau head learns to sit high.
    weight = (taus.view(1, 1, -1) - (u.detach() < 0).float()).abs()
    # Sum over OUR quantiles (each carries its own tau), mean over the
    # targets -- paper eq. 10. Swapping the two rescales the gradient by N.
    return (weight * huber / kappa).sum(2).mean(1)


class QRDQN(NStepDoubleDQN):

    def __init__(self, *args, n_quantiles=51, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_quant = n_quantiles
        self._taus = quantile_midpoints(n_quantiles, DEVICE)

    def _q(self, states):
        """The mean over quantiles IS the Q-value. (B, A)."""
        return self._estimator(states).mean(2)

    def step(self, states, masks=None):
        # stage 02's batched step, with argmax over self._q (the mean over
        # quantiles) instead of the raw output
        n = len(states)
        masks = _mask_or_ones(masks, n, self._action_num)
        if self._step_cnt < self._warmup_steps:
            return random_legal(masks)
        self.eps = next(self._epsilon)
        greedy = np.random.rand(n) >= self.eps
        actions = random_legal(masks)
        if greedy.any():
            with torch.no_grad():
                q = self._q(_to_tensor(states[greedy]))
                actions[greedy] = _apply_mask(q, masks[greedy]).argmax(1).cpu().numpy()
        return actions

    def _update(self, states, actions, rewards, states_, mask, n, legal_):
        self._legal_ = legal_
        pred, per_sample = self._per_sample(states, actions, rewards,
                                            states_, mask, n)
        loss = per_sample.mean()
        self._backward(loss)
        self.loss = loss.item()
        self.mean_q = pred.mean().item()

    def _per_sample(self, states, actions, rewards, states_, mask, n):
        """-> (pred quantiles (B,N), loss per sample (B,)).

        Stage 08 weights the second; stage 10 augments the inputs.
        """
        with torch.no_grad():
            target = self._target_quantiles(states, actions, rewards,
                                            states_, mask, n)
        idx = actions.unsqueeze(2).expand(-1, 1, self._n_quant)     # (B,1,N)
        pred = self._estimator(states).gather(1, idx).squeeze(1)    # (B,N)
        return pred, quantile_huber_loss(pred, target, self._taus)

    def _target_quantiles(self, states, actions, rewards, states_, mask, n):
        """Double-DQN target, N quantiles wide.

        Stage 07 replaces this; it needs `states` and `actions`, which this
        version ignores.
        """
        # masked before the argmax; _mask_next is a no-op when unmasked
        a_ = self._mask_next(self._q(states_)).argmax(1)         # online picks
        idx = a_.view(-1, 1, 1).expand(-1, 1, self._n_quant)
        quant_ = self._target_estimator(states_).gather(1, idx).squeeze(1)
        return rewards + (self._gamma ** n) * mask * quant_       # (B,1) over N


def extra_args(p):
    extra_05(p)
    p.add_argument('--n-quantiles', type=int, default=51)
    p.add_argument('--test-loss', action='store_true',
                   help='fit N(5,3) with the quantile loss and exit')


def build(args, capacity):
    _, buffer, extra = build_04(args, capacity)
    net = QRDuelingNet(N_ACTIONS, n_quantiles=args.n_quantiles,
                       lin_size=args.lin_size, stride=args.stride).to(DEVICE)
    return net, buffer, dict(extra, n_quantiles=args.n_quantiles)


# ------------------------------------------------------------- self-test


def test_loss(N=8, steps=3000):
    """Fit N quantiles to samples from N(mean 5, sd 3) and check them.

    Do NOT test this by checking the loss is minimised at pred == target for
    one fixed target vector. It isn't, and shouldn't be: the loss drives each
    head to a quantile of the target DISTRIBUTION, so against a single N-atom
    target the heads converge to that target's quantile spread, not to the
    target itself. Testing for identity makes you "fix" correct code.
    """
    torch.manual_seed(0)
    taus = quantile_midpoints(N)
    samples = torch.randn(4096, 64) * 3.0 + 5.0
    pred = torch.zeros(1, N, requires_grad=True)
    opt = torch.optim.Adam([pred], lr=0.05)
    for _ in range(steps):
        batch = samples[torch.randint(0, 4096, (64,))]
        opt.zero_grad()
        quantile_huber_loss(pred.expand(64, N), batch, taus).mean().backward()
        opt.step()

    got = pred[0].detach()
    want = torch.quantile(samples.reshape(-1), taus)
    err = (got - want).abs().max().item()
    mean = got.mean().item()
    print(f'tau    {"  ".join(f"{t:5.3f}" for t in taus)}')
    print(f'fitted {"  ".join(f"{v:5.2f}" for v in got)}')
    print(f'true   {"  ".join(f"{v:5.2f}" for v in want)}')
    print(f'\nmax |fitted - true| {err:.2f}   want < 0.6')
    print(f'mean of fitted      {mean:.2f}   want ~5.00 (true distribution mean)')
    # monotone taus must give monotone quantiles; a transposed loss breaks this
    mono = bool((got[1:] >= got[:-1]).all())
    print(f'monotone increasing {mono}   want True')
    ok = err < 0.6 and abs(mean - 5.0) < 0.3 and mono
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-loss' in (argv if argv is not None else sys.argv[1:]):
        return test_loss()
    return stage_02_main(argv, agent_cls=QRDQN, stage_id=STAGE_ID,
                         extra_args=extra_args, build=build)


if __name__ == '__main__':
    sys.exit(main() or 0)
