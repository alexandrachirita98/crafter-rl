"""Stage 12 -- expert demonstrations in the replay buffer.

DQfD -- Hester et al. 2018, https://arxiv.org/abs/1704.03732, section 3. This
is the stage closest to its source: the four mechanisms are the paper's, and
what we added is the two guard-rails at the bottom plus the Crafter-specific
screening of which demos are worth keeping.

Why bother. No published agent -- not Rainbow, not DreamerV2 -- ever makes an
iron pickaxe on Crafter, not once in a million steps. An agent cannot learn
from an experience it has never had, so instead of hoping it discovers the
pickaxe we put the pickaxe in its memory. The geometric-mean score is what
makes this the move worth making: turning a dead 0% into 10% is worth about
ln(11)/ln(1.005) ~ 480x more than polishing collect_wood from 50% to 51%.

The demos come from record_demos.py -- a hand-written BFS planner reading
info['semantic'], not a learned agent.

All four DQfD mechanisms, because three of them is not the method:

  1. Loaded before training starts, so they are in memory from step 0.
  2. Never overwritten -- the ring's write pointer wraps back to n_demo, not 0.
  3. A permanent priority bonus, so they keep being sampled long after the
     agent has stopped being surprised by them.
  4. The margin loss, which is the one people skip. Without it the agent learns
     what good moves look like but never that anything is bad -- every Q drifts
     upward together and the ranking never changes. Hester section 4.3 found
     removing it the single most damaging ablation.

Differences from the paper's description, forced by what stages 08-11 built:

  - The reference implementation calls buffer.add(obs=..., is_demo=True) and
    buffer.protect_prefix(). Neither exists here. Our buffer is stage 08's ring
    of six-tuples (seven with stage 11's label), so protection is a `_floor` on
    the write pointer and is_demo is an eighth field.
  - Demos are single steps; the buffer wants n-step returns. The loader runs
    the same n-step window stage 04 uses, so a demo transition is built exactly
    like an online one.
  - Our demos top out at make_stone_pickaxe (see `python record_demos.py`
    output). Iron is the bar for the result, not for these files, so the
    numbers to report from this stage are stone-tier ones.

Run:
    python stage_12_c3_demonstrations/train.py --test-demos
    python stage_12_c3_demonstrations/train.py --total-steps 2000 \
        --warmup-steps 500 --buffer-size 5000 --log-every 200
    python stage_12_c3_demonstrations/train.py --total-steps 200000 --seed 1
"""

import pathlib
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.metrics import ACHIEVEMENTS
from stage_02_dqn.train import DEVICE, N_ACTIONS, main as stage_02_main
from stage_10_drq_resets.train import random_shift
from stage_11_c1_achievement_aux.train import (AchievementLabeller,
                                               AuxDrQResets,
                                               AuxPrioritizedReplayMemory,
                                               AuxQRDuelingNet,
                                               build as build_11,
                                               extra_args as extra_11,
                                               train_aux)

STAGE_ID = 'stage_12_c3_demonstrations'


# ------------------------------------------------- 1. the buffer, guarded


class DemoReplayMemory(AuxPrioritizedReplayMemory):
    """Stage 11's buffer with a protected prefix and an is_demo flag.

    Eight-tuples now: (s, a, G, s_n, mask, n, y, is_demo).
    """

    def __init__(self, *args, demo_cap=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._floor = 0          # mechanism 2: the ring never wraps below this
        self._demo_cap = demo_cap

    def push(self, transition):
        if len(transition) == 7:            # an online seven-tuple
            transition = (*transition, False)
        self._buffer[self._ptr] = transition
        self._tree.update(self._ptr, max(self._tree.max_leaf(), 1.0))
        self._ptr += 1
        if self._ptr >= self._size:
            # The wrap goes back to _floor, not to 0. Otherwise the agent
            # slowly eats its own demonstrations and mechanism 2 quietly
            # stops existing.
            self._ptr = self._floor
        self._len = min(self._len + 1, self._size)

    def protect_prefix(self, n_demo):
        self._floor = n_demo
        self._ptr = n_demo

    def add_priority_bonus(self, n_demo, bonus):
        """Mechanism 3: a permanent floor under every demo's priority."""
        for i in range(n_demo):
            self._tree.update(i, self._tree.nodes[self._tree.size + i] + bonus)
        self._demo_bonus = bonus

    def update_priorities(self, idx, losses):
        """Re-apply the demo bonus; otherwise one update erases mechanism 3."""
        p = (np.asarray(losses) + self._eps) ** self._alpha
        idx = np.asarray(idx)
        is_demo = idx < self._floor
        p = p + is_demo * getattr(self, '_demo_bonus', 0.0)
        self._tree.update(idx, p)

    def sample(self):
        """Stage 11's sample, plus the is_demo column and the 10% cap.

        Guard-rail 1: cap demos at demo_cap of the batch. Stratified sampling
        over a prefix that carries a large priority bonus can otherwise hand
        back a batch that is nearly all demonstrations, and then the RL loss
        is not really in charge of anything.
        """
        B, total = self._batch_size, self._tree.total()
        prefix = (np.arange(B) + np.random.rand(B)) * total / B
        idx = self._tree.find(prefix)
        # An empty tree has total 0, and find() then walks into the padding
        # leaves the SumTree allocated to reach a power of two, which index
        # rows that were never written. It only bites when the n-step window
        # is still holding everything back at the end of warmup -- a short
        # smoke test -- but a None row here crashes somewhere confusing.
        idx = np.clip(idx, 0, max(self._len - 1, 0))

        if self._floor and self._len > self._floor:
            idx = self._enforce_cap(idx, B)

        p = self._tree.nodes[self._tree.size + idx] / total
        w = (self._len * p) ** -self.beta
        w = w / w.max()
        rows = [self._buffer[i] for i in idx]
        cols = self._unpack([r[:7] for r in rows])
        is_demo = torch.tensor([float(r[7]) for r in rows],
                               dtype=torch.float32).to(DEVICE)
        weights = torch.tensor(w, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        return (*cols, is_demo, idx, weights)

    def _enforce_cap(self, idx, B):
        """Replace surplus demo draws with uniform draws from the agent half."""
        max_demo = max(1, int(round(self._demo_cap * B)))
        demo_pos = np.flatnonzero(idx < self._floor)
        if len(demo_pos) <= max_demo:
            return idx
        drop = np.random.choice(demo_pos, len(demo_pos) - max_demo,
                                replace=False)
        idx = idx.copy()
        idx[drop] = np.random.randint(self._floor, self._len, len(drop))
        return idx


# ----------------------------------------------------------- 2. the loader


def load_demos(buffer, demo_dir, n_step=10, gamma=0.99, horizon=100,
               min_achievement='make_stone_pickaxe', verbose=True):
    """Mechanism 1: fill the buffer's prefix before step 0 ever happens.

    Each .npz is one episode from record_demos.py. We rebuild n-step
    transitions with stage 04's window so a demo row is indistinguishable in
    shape from an online row, and stage 11's labels so the aux head trains on
    them too.

    Episodes that never got deep are skipped. A demo where the expert died
    chopping wood teaches nothing the agent cannot find on its own, and it
    spends protected buffer space a stone-pickaxe episode wants.
    """
    demo_dir = pathlib.Path(demo_dir)
    files = sorted(demo_dir.glob('*.npz'))
    kept = skipped = 0
    rows = []

    for f in files:
        ep = np.load(f)
        if ep[f'achievement_{min_achievement}'].max() < 1:
            skipped += 1
            continue
        kept += 1
        rows += _episode_to_rows(ep, n_step, gamma, horizon)

    if len(rows) > buffer._size:
        # Protecting more than the buffer holds would leave no room for the
        # agent's own experience -- the Ross & Bagnell drift problem again,
        # reached from the other side.
        rows = rows[:buffer._size // 2]

    for row in rows:
        buffer.push(row)
    buffer.protect_prefix(len(rows))

    if verbose:
        print(f'demos: {kept} episodes kept, {skipped} skipped '
              f'(need {min_achievement}), {len(rows):,} transitions protected '
              f'= {100 * len(rows) / buffer._size:.1f}% of the buffer',
              flush=True)
    return len(rows)


def _episode_to_rows(ep, n_step, gamma, horizon):
    """One .npz -> a list of eight-tuples, via the stage-04 n-step window."""
    images = ep['image']
    actions = ep['action']
    rewards = ep['reward']
    discounts = ep['discount']
    counts = np.stack([ep[f'achievement_{n}'] for n in ACHIEVEMENTS], axis=1)

    labeller = AchievementLabeller(horizon=horizon)
    window = deque()
    out = []

    def emit():
        G = sum(gamma ** k * r for k, (_, _, r, _, _) in enumerate(window))
        s0, a0 = window[0][0], window[0][1]
        _, _, _, s_n, mask_n = window[-1]
        payload = (s0, a0, G, s_n, mask_n, len(window))
        window.popleft()
        return payload

    # The recorder stores T+1 frames for T actions: index 0 is the reset
    # observation, so transition t is image[t] -> image[t+1].
    for t in range(len(actions) - 1):
        window.append((images[t], int(actions[t]), float(rewards[t]),
                       images[t + 1], float(discounts[t])))
        emitted = []
        if discounts[t] == 0.0:
            while window:
                emitted.append(emit())
        elif len(window) == n_step:
            emitted.append(emit())
        for payload in emitted:
            for p, y in labeller.push(payload, counts[t]):
                out.append((*p, y, True))
    while window:
        payload = emit()
        for p, y in labeller.push(payload, counts[-1]):
            out.append((*p, y, True))
    for p, y in labeller.flush():
        out.append((*p, y, True))
    return out


# ------------------------------------------------- 3. the margin loss


def margin_loss(q_values, demo_actions, is_demo, margin=0.8):
    """Hester eq. 3, on demo transitions only.

        L = max_a [ Q(s,a) + l(a_E,a) ] - Q(s,a_E),   l = 0 on a_E else margin

    Zero when the expert's action already leads by more than the margin, large
    when the expert's action is ranked worst. That gap is the point: it teaches
    that the other actions are worse, not just that this one is good.
    """
    l = torch.full_like(q_values, margin)
    l.scatter_(1, demo_actions, 0.0)
    violation = ((q_values + l).max(dim=1).values
                 - q_values.gather(1, demo_actions).squeeze(1))
    return (violation * is_demo).sum() / is_demo.sum().clamp(min=1)


def lambda_e(step, lam, fade_start, fade_steps):
    """Guard-rail 2: fade the imitation pressure after fade_start.

    Ross & Bagnell 2010 Theorem 2.1 -- an imitator that drifts off the
    demonstrated distribution compounds its own errors quadratically in the
    horizon, and the cure is handing control back to the RL loss. Both numbers
    are ours rather than the paper's, and untuned.
    """
    if step < fade_start:
        return lam
    return lam * max(0.0, 1.0 - (step - fade_start) / fade_steps)


# ----------------------------------------------------------- 4. the agent


class DQfD(AuxDrQResets):

    def __init__(self, *args, demo_margin=0.8, lambda_e=1.0,
                 demo_fade_start=200_000, demo_fade_steps=200_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._margin = demo_margin
        self._lambda_e = lambda_e
        self._fade_start = demo_fade_start
        self._fade_steps = demo_fade_steps
        self.margin_loss_val = float('nan')
        self.demo_frac = float('nan')

    def _update(self, states, actions, rewards, states_, mask, n, labels,
                is_demo, idx, weights):
        from common.replay import beta_by_step
        self._buffer.beta = beta_by_step(self._step_cnt, self._total_steps)

        pred, per_sample = self._per_sample(states, actions, rewards,
                                            states_, mask, n)
        rl_loss = (weights.squeeze(1) * per_sample).mean()

        shifted = random_shift(states)
        aux_loss = F.cross_entropy(self._estimator.aux_logits(shifted), labels,
                                   weight=self._aux_weight)

        lam_e = lambda_e(self._step_cnt, self._lambda_e, self._fade_start,
                         self._fade_steps)
        if lam_e > 0 and is_demo.sum() > 0:
            # The margin ranks actions, so it works on the mean over
            # quantiles -- the same Q the agent acts on.
            q = self._estimator(states).mean(2)
            m_loss = margin_loss(q, actions, is_demo, self._margin)
        else:
            m_loss = torch.zeros((), device=DEVICE)

        (rl_loss + self._lambda_aux * aux_loss + lam_e * m_loss).backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._buffer.update_priorities(idx, per_sample.detach().cpu().numpy())
        self.loss = rl_loss.item()
        self.mean_q = pred.mean().item()
        self.aux_loss = aux_loss.item()
        self.margin_loss_val = m_loss.item()
        self.demo_frac = is_demo.mean().item()
        with torch.no_grad():
            logits = self._estimator.aux_logits(shifted)
            self.aux_acc = (logits.argmax(1) == labels).float().mean().item()
            from stage_11_c1_achievement_aux.train import NONE_SOON
            self.aux_major = (labels == NONE_SOON).float().mean().item()


def extra_args(p):
    extra_11(p)
    p.add_argument('--demo-dir', type=str, default='demos')
    p.add_argument('--no-demos', action='store_true',
                   help='the control arm: identical code, empty prefix')
    p.add_argument('--demo-min-achievement', type=str,
                   default='make_stone_pickaxe',
                   help='skip demo episodes that never reached this')
    p.add_argument('--demo-priority-bonus', type=float, default=1.0,
                   help="eps_d in Hester et al.; the paper's value")
    p.add_argument('--demo-cap', type=float, default=0.1,
                   help='max fraction of a batch that may be demos')
    p.add_argument('--demo-margin', type=float, default=0.8)
    p.add_argument('--lambda-e', type=float, default=1.0,
                   help='margin-loss weight before the fade')
    p.add_argument('--demo-fade-start', type=int, default=200_000)
    p.add_argument('--demo-fade-steps', type=int, default=200_000)
    p.add_argument('--test-demos', action='store_true',
                   help='check the margin, the prefix and the cap, then exit')


def build(args, capacity):
    _, _, extra = build_11(args, capacity)
    net = AuxQRDuelingNet(N_ACTIONS, n_quantiles=args.n_quantiles,
                          lin_size=args.lin_size, stride=args.stride).to(DEVICE)
    buffer = DemoReplayMemory(size=capacity, batch_size=args.batch_size,
                              alpha=args.per_alpha, demo_cap=args.demo_cap)

    if not args.no_demos:
        n_demo = load_demos(buffer, args.demo_dir, n_step=args.n_step,
                            gamma=args.gamma, horizon=args.aux_horizon,
                            min_achievement=args.demo_min_achievement)
        if n_demo:
            buffer.add_priority_bonus(n_demo, args.demo_priority_bonus)
    else:
        print('demos: DISABLED (--no-demos), this is the control arm',
              flush=True)

    return net, buffer, dict(extra, demo_margin=args.demo_margin,
                             lambda_e=args.lambda_e,
                             demo_fade_start=args.demo_fade_start,
                             demo_fade_steps=args.demo_fade_steps)


# ------------------------------------------------------------- self-test


def test_demos():
    """Four checks, one per mechanism. All four fail silently if wrong."""
    ok = True

    # -- mechanism 4: the margin, on hand-made Q-values -------------------
    q = torch.tensor([[5.0, 1.0, 1.0],        # expert leads by 4 > 0.8
                      [1.0, 1.0, 5.0]])       # expert is worst by 4
    a = torch.tensor([[0], [0]])
    both = torch.tensor([1.0, 1.0])
    lead = margin_loss(q[:1], a[:1], both[:1]).item()
    worst = margin_loss(q[1:], a[1:], both[1:]).item()
    print('margin loss')
    print(f'  expert action leads by 4   {lead:.2f}   want 0.00')
    print(f'  expert action worst by 4   {worst:.2f}   want 4.80 (4 + 0.8)')
    ok &= abs(lead) < 1e-6 and abs(worst - 4.8) < 1e-6

    # It must ignore agent transitions entirely.
    none = margin_loss(q[1:], a[1:], torch.tensor([0.0])).item()
    print(f'  is_demo all zero           {none:.2f}   want 0.00')
    ok &= abs(none) < 1e-6

    # -- guard-rail 2: the fade ------------------------------------------
    print('\nlambda_e fade, start 200k over 200k steps')
    for step, want in ((0, 1.0), (199_999, 1.0), (300_000, 0.5),
                       (400_000, 0.0), (900_000, 0.0)):
        got = lambda_e(step, 1.0, 200_000, 200_000)
        good = abs(got - want) < 1e-9
        ok &= good
        print(f'  step {step:>7,}  {got:4.2f}   want {want:4.2f}  '
              f'{"ok" if good else "WRONG"}')

    # -- mechanisms 1-3 and the cap, on the real demo files ---------------
    demo_dir = pathlib.Path(__file__).resolve().parents[1] / 'demos'
    if not demo_dir.exists() or not list(demo_dir.glob('*.npz')):
        print(f'\nno demos in {demo_dir}; run record_demos.py first')
        print('\nFAIL')
        return 1

    np.random.seed(0)
    capacity, batch = 4000, 32
    buffer = DemoReplayMemory(size=capacity, batch_size=batch, alpha=0.5,
                              demo_cap=0.1)
    n_demo = load_demos(buffer, demo_dir, n_step=3, gamma=0.99, horizon=20)
    buffer.add_priority_bonus(n_demo, 1.0)

    loaded = n_demo > 0
    all_flagged = all(buffer._buffer[i][7] for i in range(n_demo))
    print(f'\nmechanism 1: loaded before training    {loaded}   want True  '
          f'({n_demo:,} transitions)')
    print(f'mechanism 4: every prefix row flagged  {all_flagged}   want True')
    ok &= loaded and all_flagged

    # Mechanism 2: push past the end of the ring and check that nothing in
    # the prefix moved.
    head = [buffer._buffer[i] for i in range(min(5, n_demo))]
    dummy = (np.zeros((64, 64, 3), np.uint8), 0, 0.0,
             np.zeros((64, 64, 3), np.uint8), 1.0, 1, 22)
    for _ in range(capacity + 50):
        buffer.push(dummy)
    intact = all(buffer._buffer[i][7] for i in range(n_demo))
    same = all(buffer._buffer[i][1] == head[i][1] for i in range(len(head)))
    wrapped = buffer._ptr >= n_demo
    print(f'mechanism 2: prefix survives the wrap  {intact and same}   '
          f'want True  (ptr {buffer._ptr}, floor {buffer._floor})')
    ok &= intact and same and wrapped

    # Mechanism 3 + guard-rail 1: demos keep being drawn, but capped.
    fracs = []
    for _ in range(30):
        out = buffer.sample()
        fracs.append(out[7].mean().item())
    mean_frac = float(np.mean(fracs))
    max_frac = float(np.max(fracs))
    sampled = mean_frac > 0
    capped = max_frac <= 0.1 + 1e-9 + 1.0 / batch
    print(f'mechanism 3: demos still sampled       {sampled}   want True  '
          f'(mean {mean_frac:.1%} of each batch)')
    print(f'guard-rail 1: capped at 10%            {capped}   want True  '
          f'(max {max_frac:.1%})')
    ok &= sampled and capped

    # The bonus has to survive a priority update, or mechanism 3 lasts
    # exactly one step.
    idx = np.arange(min(8, n_demo))
    before = buffer._tree.nodes[buffer._tree.size + idx].copy()
    buffer.update_priorities(idx, np.zeros(len(idx)))
    after = buffer._tree.nodes[buffer._tree.size + idx]
    survives = bool((after >= 1.0).all())
    print(f'mechanism 3: bonus survives an update  {survives}   want True  '
          f'({before.mean():.2f} -> {after.mean():.2f}, floor 1.0)')
    ok &= survives

    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def main(argv=None):
    if '--test-demos' in (argv if argv is not None else sys.argv[1:]):
        return test_demos()
    import stage_02_dqn.train as s02
    original = s02.train
    s02.train = train_aux
    try:
        return stage_02_main(argv, agent_cls=DQfD, stage_id=STAGE_ID,
                             extra_args=extra_args, build=build)
    finally:
        s02.train = original


if __name__ == '__main__':
    sys.exit(main() or 0)
