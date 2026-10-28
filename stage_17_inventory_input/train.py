"""Stage 17 -- an inventory side-input (docs/beyond-c1-c2-c3.md §4, #5).

The agent sees one 64x64 frame. Its inventory and vitals are in there, but
only as small HUD icons, and stage 02's encoder has a 5x5 receptive field --
it has to count icons with a net that cannot count. Almost every decision in
Crafter is inventory-gated, and the champion makes a wood pickaxe in 18% of
episodes and then barely uses it.

--inv-input concatenates the 16 inventory/vital counts onto the features
before the head, normalised by their data.yaml max of 9. --no-inv-input is
the control arm: same code path, same parameter count, zeroed side-input.

Chosen over the doc's other option, frame stacking, because 4 frames would
force --buffer-size down from 500000 and so confound the comparison.

Privileged state, more so than the mask: the mask only removed actions that
provably did nothing, this hands over state the agent was meant to infer
from pixels. Report it against our own baseline, not the Crafter paper's.

The side-input has to reach the net when acting (one state) and when
learning (a batch of states and of next states), and s_'s inventory is not
s's -- so both go in the replay tuple. Pairing them wrong trains the target
on the wrong state's inventory without raising anything.

Run:
    python stage_17_inventory_input/train.py --test-invinput
    python stage_17_inventory_input/train.py --total-steps 200000 --seed 1 \
        --num-envs 16 --mask --seed-stride 16 --inv-input
"""

import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from common.env import N_ACTIONS, OBS_SHAPE
from stage_02_dqn.train import (DEVICE, ByteToFloat, DQN, ReplayMemory, View,
                                _apply_mask, _to_tensor,
                                main as stage_02_main)

STAGE_ID = 'stage_17_inventory_input'

# crafter/data.yaml items:. The order is baked into the net's input layer,
# so it is frozen here rather than taken from a dict's iteration order.
INV_KEYS = ('health', 'food', 'drink', 'energy',
            'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond',
            'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
            'wood_sword', 'stone_sword', 'iron_sword')
N_INV = len(INV_KEYS)

# data.yaml gives all of them max 9, vitals included.
INV_MAX = 9.0


def inv_vector(info):
    """info -> (16,) float32 in [0, 1]. Zeros when info has no inventory,
    which is also what the control arm feeds."""
    out = np.zeros(N_INV, dtype=np.float32)
    inv = (info or {}).get('inventory')
    if inv is None:
        return out
    for k, key in enumerate(INV_KEYS):
        out[k] = inv.get(key, 0) / INV_MAX
    return out


class InvQNet(nn.Module):
    """Stage 02's conv stack with the inventory joined on before the head.

    The trunk matches get_estimator exactly, so --no-inv-input reproduces the
    champion's architecture with a 16-wide block of zeros in place.
    """

    def __init__(self, n_actions=N_ACTIONS, input_ch=OBS_SHAPE[2],
                 lin_size=64, stride=1, use_inv=True):
        super().__init__()
        self.use_inv = use_inv
        self.conv = nn.Sequential(
            ByteToFloat(),
            nn.Conv2d(input_ch, 16, kernel_size=3, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2, stride=stride),
            nn.ReLU(inplace=True),
            View(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, input_ch, OBS_SHAPE[0], OBS_SHAPE[1],
                                dtype=torch.uint8)
            flat = self.conv(probe).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, lin_size),
                                nn.ReLU(inplace=True))
        # N_INV extra inputs either way, so both arms have the same count.
        self.head = nn.Linear(lin_size + N_INV, n_actions)

    def forward(self, obs, inv=None):
        """obs: NCHW uint8. inv: (B, 16) float, or None for zeros."""
        f = self.fc(self.conv(obs))
        if inv is None or not self.use_inv:
            inv = torch.zeros(f.shape[0], N_INV, device=f.device,
                              dtype=f.dtype)
        return self.head(torch.cat([f, inv], dim=1))


class InvReplayMemory(ReplayMemory):
    """Stage 02's deque, eight-tuples: (s, a, r, s_, mask, legal_, inv, inv_).

    inv belongs to s and inv_ to s_; they differ, so they are stored apart.
    """

    def sample(self):
        s, a, r, s_, m, lg, iv, iv_ = zip(
            *random.sample(self._buffer, self._batch_size))
        return (
            _to_tensor(np.stack(s)),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            _to_tensor(np.stack(s_)),
            torch.tensor(m, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.from_numpy(np.stack(lg)).to(DEVICE),
            torch.from_numpy(np.stack(iv)).to(DEVICE),
            torch.from_numpy(np.stack(iv_)).to(DEVICE),
        )


class InvDoubleDQN(DQN):
    """Double DQN whose every net call carries the matching inventory."""

    def __init__(self, *args, use_inv=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_inv = use_inv
        # Set by train() before each step(). Held here rather than passed in,
        # because step(states, masks) is the signature every stage shares.
        self._cur_inv = np.zeros((self._num_envs, N_INV), dtype=np.float32)

    def step(self, states, masks=None):
        """Epsilon-greedy, with the inventory alongside the frames."""
        from stage_02_dqn.train import _mask_or_ones, random_legal

        n = len(states)
        masks = _mask_or_ones(masks, n, self._action_num)
        if self._step_cnt < self._warmup_steps:
            return random_legal(masks)

        self.eps = next(self._epsilon)
        greedy = np.random.rand(n) >= self.eps
        actions = random_legal(masks)
        if greedy.any():
            with torch.no_grad():
                inv = torch.from_numpy(self._cur_inv[greedy]).to(DEVICE)
                qvals = self._estimator(_to_tensor(states[greedy]), inv)
                qvals = _apply_mask(qvals, masks[greedy])
                actions[greedy] = qvals.argmax(1).cpu().numpy()
        return actions

    def _store(self, i, state, action, reward, state_, mask, legal_,
               inv=None, inv_=None):
        if inv is None:
            inv = np.zeros(N_INV, dtype=np.float32)
        if inv_ is None:
            inv_ = np.zeros(N_INV, dtype=np.float32)
        self._buffer.push((state, action, reward, state_, mask, legal_,
                           inv, inv_))

    def learn(self, i, state, action, reward, state_, mask, legal_=None,
              inv=None, inv_=None):
        if legal_ is None:
            legal_ = np.ones(self._action_num, dtype=np.float32)
        self._store(i, state, action, reward, state_, mask, legal_, inv, inv_)

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return
        if self._step_cnt % self._update_steps == 0:
            self._update(*self._buffer.sample())
        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(
                self._estimator.state_dict())
        self._step_cnt += 1

    def _update(self, states, actions, rewards, states_, mask, legal_,
                inv, inv_):
        with torch.no_grad():
            # inv_ with states_, on both nets: inv here would be the silent
            # bug test_invinput checks for.
            actions_ = _apply_mask(self._estimator(states_, inv_),
                                   legal_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_, inv_)
        q_values = self._estimator(states, inv)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)
        target_qsa = rewards + self._gamma * qsa_ * mask

        loss = self._td_loss(qsa, target_qsa)
        self._backward(loss)
        self.loss = loss.item()
        self.mean_q = qsa.mean().item()


def train(agent, env, step_num=100_000, log_every=10_000,
          use_mask=False, mask_extended=False, mask_skip_grass=False,
          ckpt_path=None, ckpt_every=0, start_step=0, start_episodes=0):
    """Stage 02's loop, carrying the inventory vectors with the frames."""
    import signal
    import time
    from collections import deque

    from common.env import bootstrap_mask
    from common.masking import legal_masks
    from stage_02_dqn.train import save_ckpt

    num_envs = env.num_envs
    ep_reward = np.zeros(num_envs, dtype=np.float64)
    recent = deque(maxlen=10)
    states = env.reset()
    episodes = start_episodes
    start = time.time()

    dying = {'now': False}

    def _on_term(signum, frame):
        del signum, frame
        dying['now'] = True

    if ckpt_path is not None:
        signal.signal(signal.SIGTERM, _on_term)
    masks = None
    done_steps = start_step
    # reset() returns no info, so the first step acts on zeros.
    cur_inv = np.zeros((num_envs, N_INV), dtype=np.float32)

    for step in range(start_step, step_num, num_envs):
        agent._cur_inv = cur_inv
        actions = agent.step(states, masks)
        assert np.shape(actions) == (num_envs,), (
            f'{type(agent).__name__}.step() returned {np.shape(actions)}, '
            f'want ({num_envs},) -- a single-env step() on a vec env')

        states_, rewards, dones, infos = env.step(actions)

        if use_mask:
            masks = legal_masks(infos, dones, extended=mask_extended,
                                skip_grass=mask_skip_grass)

        # On a done step infos[i] is the dying step's, which is the right
        # inventory for the final transition; mask=0 means it is never
        # bootstrapped off anyway.
        next_inv = np.stack([inv_vector(info) for info in infos])

        for i in range(num_envs):
            next_state = states_[i]
            if dones[i] and 'final_obs' in infos[i]:
                next_state = infos[i]['final_obs']
            agent.learn(i, states[i], actions[i], rewards[i], next_state,
                        bootstrap_mask(dones[i], infos[i]),
                        masks[i] if use_mask else None,
                        inv=cur_inv[i], inv_=next_inv[i])
            ep_reward[i] += rewards[i]
            if dones[i]:
                agent.end_episode(i)
                recent.append(ep_reward[i])
                ep_reward[i] = 0.0
                episodes += 1

        states = states_
        # Next step's current inventory is this step's next, except after a
        # done where the env has reset: zeros there, as at reset.
        cur_inv = next_inv.copy()
        for i in range(num_envs):
            if dones[i]:
                cur_inv[i] = 0.0

        done_steps = step + num_envs
        if done_steps % log_every < num_envs:
            fps = done_steps / (time.time() - start)
            seen = list(recent) or [float('nan')]
            print('[{0:4d}][{1:8d}], R/ep={2:6.2f}, eps={3:4.2f}, '
                  'loss={4:8.4f}, meanQ={5:7.3f}, {6:5.0f} fps'.format(
                      episodes, done_steps,
                      float(np.mean(seen)), agent.eps,
                      agent.loss, agent.mean_q, fps), flush=True)

        if ckpt_path is not None and ckpt_every \
                and done_steps % ckpt_every < num_envs:
            save_ckpt(ckpt_path, agent, done_steps, episodes)

        if dying['now']:
            if ckpt_path is not None:
                save_ckpt(ckpt_path, agent, done_steps, episodes)
            print(f'\nSIGTERM at {done_steps:,} steps -- checkpointed',
                  flush=True)
            break

    return done_steps, episodes, time.time() - start


def extra_args(p):
    inv = p.add_mutually_exclusive_group()
    inv.add_argument('--inv-input', dest='inv_input', action='store_true',
                     help='feed the 16 inventory/vital counts to the net')
    inv.add_argument('--no-inv-input', dest='inv_input', action='store_false',
                     help='zeroed side-input: the control arm')
    p.set_defaults(inv_input=False)
    p.add_argument('--test-invinput', action='store_true',
                   help='check the net, the buffer alignment, then exit')


def build(args, capacity):
    net = InvQNet(N_ACTIONS, lin_size=args.lin_size, stride=args.stride,
                  use_inv=args.inv_input).to(DEVICE)
    buffer = InvReplayMemory(size=capacity, batch_size=args.batch_size)
    print(f'inventory side-input: {"ON" if args.inv_input else "OFF (control)"}')
    return net, buffer, dict(use_inv=args.inv_input)


# ------------------------------------------------------------- self-test


def test_invinput():
    ok = True

    # -- inv_vector: fixed order, scaled to [0, 1], zeros on empty --------
    info = {'inventory': {k: 0 for k in INV_KEYS}}
    info['inventory']['wood'] = 9
    info['inventory']['health'] = 9
    v = inv_vector(info)
    if v.shape != (N_INV,) or v.dtype != np.float32:
        print(f'FAIL inv_vector shape/dtype: {v.shape} {v.dtype}'); ok = False
    if abs(v[INV_KEYS.index('wood')] - 1.0) > 1e-6:
        print('FAIL wood=9 should normalise to 1.0'); ok = False
    if v.max() > 1.0 or v.min() < 0.0:
        print('FAIL inv_vector must lie in [0, 1]'); ok = False
    if inv_vector(None).any() or inv_vector({}).any():
        print('FAIL missing inventory should give zeros'); ok = False

    # -- the net takes the side-input and uses it -------------------------
    batch = 4
    obs = torch.randint(0, 256, (batch, 3, 64, 64), dtype=torch.uint8).to(DEVICE)
    net = InvQNet(lin_size=64, use_inv=True).to(DEVICE)
    a = torch.zeros(batch, N_INV, device=DEVICE)
    b = torch.ones(batch, N_INV, device=DEVICE)
    with torch.no_grad():
        qa, qb = net(obs, a), net(obs, b)
    if qa.shape != (batch, N_ACTIONS):
        print(f'FAIL net output {tuple(qa.shape)}'); ok = False
    if torch.allclose(qa, qb, atol=1e-6):
        print('FAIL the inventory changes nothing -- side-input not wired')
        ok = False

    # -- the control arm must ignore it ------------------------------------
    off = InvQNet(lin_size=64, use_inv=False).to(DEVICE)
    with torch.no_grad():
        oa, ob = off(obs, a), off(obs, b)
    if not torch.allclose(oa, ob, atol=1e-6):
        print('FAIL --no-inv-input must ignore the side-input'); ok = False

    # -- buffer alignment: inv_ belongs to s_, not to s --------------------
    buf = InvReplayMemory(size=16, batch_size=2)
    for t in range(4):
        s = np.full((64, 64, 3), t, dtype=np.uint8)
        s_ = np.full((64, 64, 3), t + 1, dtype=np.uint8)
        iv = np.full(N_INV, t / 10.0, dtype=np.float32)
        iv_ = np.full(N_INV, (t + 1) / 10.0, dtype=np.float32)
        buf.push((s, 0, 0.0, s_, 1.0,
                  np.ones(N_ACTIONS, dtype=np.float32), iv, iv_))
    out = buf.sample()
    if len(out) != 8:
        print(f'FAIL sample() returned {len(out)} items, want 8'); ok = False
    states, _, _, states_, _, _, iv, iv_ = out
    # Each row's frame value t pairs with inv t/10 and next-inv (t+1)/10.
    for r in range(2):
        t = int(states[r, 0, 0, 0].item())
        if abs(iv[r, 0].item() - t / 10.0) > 1e-5:
            print(f'FAIL inv not aligned with s on row {r}'); ok = False
        if abs(iv_[r, 0].item() - (t + 1) / 10.0) > 1e-5:
            print(f'FAIL inv_ not aligned with s_ on row {r}'); ok = False
        if int(states_[r, 0, 0, 0].item()) != t + 1:
            print(f'FAIL s_ not aligned on row {r}'); ok = False

    print('inventory side-input self-test:', 'OK' if ok else 'FAILED')
    return ok


def main(argv=None):
    if '--test-invinput' in (argv if argv is not None else sys.argv[1:]):
        raise SystemExit(0 if test_invinput() else 1)
    import stage_02_dqn.train as s02
    original = s02.train
    s02.train = train
    try:
        return stage_02_main(argv, agent_cls=InvDoubleDQN, stage_id=STAGE_ID,
                             extra_args=extra_args, build=build)
    finally:
        s02.train = original


if __name__ == '__main__':
    main()
