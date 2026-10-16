"""PPO baseline for Crafter, written by a coding agent (not hand-written).

Runnable bare:  python baseline_agent/train.py
Smoke test:     python baseline_agent/train.py --steps 2000 --outdir /tmp/smoke

Plain PyTorch only. No stable-baselines3, rllib or other DRL library.
"""

import argparse
import collections
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crafter


# --------------------------------------------------------------------------
# Environment plumbing
# --------------------------------------------------------------------------

def make_env(outdir, seed):
  """A single Crafter env, wrapped so episode stats land in stats.jsonl."""
  env = crafter.Env(seed=seed)
  env = crafter.Recorder(
      env, outdir,
      save_stats=True,
      save_video=False,
      save_episode=False,
  )
  return env


class VecEnv:
  """Synchronous vector of Crafter envs with auto-reset.

  Every env gets its OWN Recorder writing to <outdir>/shardN/stats.jsonl. Two
  reasons this layout and not a single shared file: appending from 16 envs to
  one handle interleaves partial lines, and analysis/read_metrics.py globs
  '**/stats.jsonl' treating each file as one run -- which is also how the
  random-agent runs in logdir/ are already laid out. Crafter's step() returns a
  4-tuple (obs, reward, done, info) -- old gym -- so we match that, not
  gymnasium's 5-tuple.
  """

  def __init__(self, num_envs, outdir, seed):
    self.envs = [
        make_env(f'{outdir}/shard{i}' if outdir else None, seed + i)
        for i in range(num_envs)
    ]
    self.num_envs = num_envs
    self.num_actions = self.envs[0].action_space.n

  def reset(self):
    return np.stack([env.reset() for env in self.envs])

  def step(self, actions):
    obs, rewards, dones, infos = [], [], [], []
    for env, action in zip(self.envs, actions):
      ob, reward, done, info = env.step(int(action))
      if done:
        # Auto-reset. The observation we return is the first of the next
        # episode; `done` tells the learner not to bootstrap across the seam.
        ob = env.reset()
      obs.append(ob)
      rewards.append(reward)
      dones.append(done)
      infos.append(info)
    return np.stack(obs), np.array(rewards, np.float32), np.array(dones), infos


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias)
  return layer


class ActorCritic(nn.Module):
  """Nature-CNN sized down for 64x64 input, with separate policy/value heads."""

  def __init__(self, num_actions):
    super().__init__()
    self.encoder = nn.Sequential(
        layer_init(nn.Conv2d(3, 32, 8, stride=4)),   # 64 -> 15
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # 15 -> 6
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # 6 -> 4
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 4 * 4, 512)),
        nn.ReLU(),
    )
    # Small policy gain keeps the initial policy near-uniform over 17 actions.
    self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
    self.critic = layer_init(nn.Linear(512, 1), std=1.0)

  def features(self, obs_uint8):
    # Observations are stored as uint8 everywhere; this is the ONE place they
    # become floats and the only place they are divided by 255.
    x = obs_uint8.float().div_(255.0).permute(0, 3, 1, 2)
    return self.encoder(x)

  def value(self, obs_uint8):
    return self.critic(self.features(obs_uint8)).squeeze(-1)

  def act(self, obs_uint8):
    feat = self.features(obs_uint8)
    logits = self.actor(feat)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action, dist.log_prob(action), self.critic(feat).squeeze(-1)

  def evaluate(self, obs_uint8, actions):
    feat = self.features(obs_uint8)
    dist = torch.distributions.Categorical(logits=self.actor(feat))
    return (
        dist.log_prob(actions),
        dist.entropy(),
        self.critic(feat).squeeze(-1),
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', default='logdir/crafter_reward-baseline/0')
  parser.add_argument('--steps', type=float, default=1e6,
                      help='budget in ENVIRONMENT steps')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--num-envs', type=int, default=16)
  parser.add_argument('--horizon', type=int, default=64,
                      help='steps per env per rollout')
  parser.add_argument('--epochs', type=int, default=3)
  parser.add_argument('--minibatches', type=int, default=4)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--gamma', type=float, default=0.95)
  parser.add_argument('--gae-lambda', type=float, default=0.65)
  parser.add_argument('--clip-coef', type=float, default=0.2)
  parser.add_argument('--entropy-coef', type=float, default=0.01)
  parser.add_argument('--value-coef', type=float, default=0.5)
  parser.add_argument('--max-grad-norm', type=float, default=0.5)
  parser.add_argument('--log-every', type=int, default=10_000,
                      help='env steps between progress lines')
  parser.add_argument('--ckpt-every', type=int, default=100_000,
                      help='env steps between checkpoints')
  return parser.parse_args()


def compute_gae(rewards, values, dones, last_value, gamma, lam):
  """Truncated GAE(lambda).

  `dones[t]` marks that the transition at t ended the episode, so the value of
  the state reached at t+1 belongs to a FRESH episode and must not be
  bootstrapped. `nextnonterminal` zeroes both the bootstrap and the advantage
  carry at every episode boundary.
  """
  horizon = rewards.shape[0]
  advantages = torch.zeros_like(rewards)
  lastgaelam = torch.zeros_like(last_value)
  for t in reversed(range(horizon)):
    if t == horizon - 1:
      nextnonterminal = 1.0 - dones[t]
      nextvalue = last_value
    else:
      nextnonterminal = 1.0 - dones[t]
      nextvalue = values[t + 1]
    delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
    lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    advantages[t] = lastgaelam
  return advantages, advantages + values


def main():
  args = parse_args()
  total_steps = int(args.steps)
  outdir = pathlib.Path(args.outdir).expanduser()
  outdir.mkdir(parents=True, exist_ok=True)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  device = torch.device(args.device)
  envs = VecEnv(args.num_envs, str(outdir), args.seed)
  agent = ActorCritic(envs.num_actions).to(device)
  optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

  batch_size = args.num_envs * args.horizon
  minibatch_size = batch_size // args.minibatches

  print(f'device={device} envs={args.num_envs} horizon={args.horizon} '
        f'batch={batch_size} actions={envs.num_actions}')
  print(f'budget={total_steps} env steps -> {total_steps // batch_size} updates')

  # Rollout storage. Observations stay uint8 (16 x 64 x 64 x 64 x 3 = 12 MB);
  # they are cast to float only inside ActorCritic.features().
  obs_buf = torch.zeros(
      (args.horizon, args.num_envs, 64, 64, 3), dtype=torch.uint8, device=device)
  act_buf = torch.zeros((args.horizon, args.num_envs), dtype=torch.long, device=device)
  logp_buf = torch.zeros((args.horizon, args.num_envs), device=device)
  rew_buf = torch.zeros((args.horizon, args.num_envs), device=device)
  done_buf = torch.zeros((args.horizon, args.num_envs), device=device)
  val_buf = torch.zeros((args.horizon, args.num_envs), device=device)

  next_obs = torch.as_tensor(envs.reset(), device=device)
  next_done = torch.zeros(args.num_envs, device=device)

  # Running stats over the most recent finished episodes, for the log line.
  ep_rewards = collections.deque(maxlen=100)
  ep_lengths = collections.deque(maxlen=100)
  cur_reward = np.zeros(args.num_envs, np.float32)
  cur_length = np.zeros(args.num_envs, np.int64)

  env_steps = 0            # ENVIRONMENT steps -- the budget is spent in these
  next_log = args.log_every
  next_ckpt = args.ckpt_every
  start = time.time()
  last_log_time, last_log_steps = start, 0

  while env_steps < total_steps:
    # ---- collect a rollout ------------------------------------------------
    for t in range(args.horizon):
      obs_buf[t] = next_obs
      done_buf[t] = next_done
      with torch.no_grad():
        action, logp, value = agent.act(next_obs)
      act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

      obs, reward, done, _ = envs.step(action.cpu().numpy())
      env_steps += envs.num_envs

      cur_reward += reward
      cur_length += 1
      for i in range(envs.num_envs):
        if done[i]:
          ep_rewards.append(float(cur_reward[i]))
          ep_lengths.append(int(cur_length[i]))
          cur_reward[i] = 0.0
          cur_length[i] = 0

      rew_buf[t] = torch.as_tensor(reward, device=device)
      next_obs = torch.as_tensor(obs, device=device)
      next_done = torch.as_tensor(done.astype(np.float32), device=device)

    # done_buf[t] is the done flag of the transition AT t, needed by GAE.
    dones_at_t = torch.cat([done_buf[1:], next_done.unsqueeze(0)], 0)

    with torch.no_grad():
      last_value = agent.value(next_obs)
      advantages, returns = compute_gae(
          rew_buf, val_buf, dones_at_t, last_value, args.gamma, args.gae_lambda)

    # ---- PPO update ------------------------------------------------------
    b_obs = obs_buf.reshape(-1, 64, 64, 3)
    b_act = act_buf.reshape(-1)
    b_logp = logp_buf.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_ret = returns.reshape(-1)

    indices = np.arange(batch_size)
    for _ in range(args.epochs):
      np.random.shuffle(indices)
      for start_idx in range(0, batch_size, minibatch_size):
        mb = indices[start_idx:start_idx + minibatch_size]
        new_logp, entropy, new_value = agent.evaluate(b_obs[mb], b_act[mb])

        ratio = (new_logp - b_logp[mb]).exp()
        mb_adv = b_adv[mb]
        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # Clipped surrogate (maximised -> negated below).
        pg_loss = torch.max(
            -mb_adv * ratio,
            -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
        ).mean()
        value_loss = F.mse_loss(new_value, b_ret[mb])
        entropy_loss = entropy.mean()

        loss = (pg_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

    # ---- logging and checkpoints -----------------------------------------
    if env_steps >= next_log:
      now = time.time()
      fps = (env_steps - last_log_steps) / max(now - last_log_time, 1e-6)
      mean_reward = np.mean(ep_rewards) if ep_rewards else float('nan')
      mean_length = np.mean(ep_lengths) if ep_lengths else float('nan')
      print(f'step {env_steps:>8d}  reward {mean_reward:6.2f}  '
            f'length {mean_length:6.1f}  episodes {len(ep_rewards):3d}  '
            f'fps {fps:6.0f}  elapsed {now - start:6.0f}s', flush=True)
      last_log_time, last_log_steps = now, env_steps
      while next_log <= env_steps:
        next_log += args.log_every

    if env_steps >= next_ckpt:
      torch.save({'step': env_steps, 'model': agent.state_dict()},
                 outdir / 'ckpt.pt')
      while next_ckpt <= env_steps:
        next_ckpt += args.ckpt_every

  torch.save({'step': env_steps, 'model': agent.state_dict()},
             outdir / 'ckpt.pt')
  print(f'done: {env_steps} env steps in {time.time() - start:.0f}s '
        f'-> {outdir}')


if __name__ == '__main__':
  main()
