"""Crafter environment construction.

One place, so that every stage sees exactly the same environment. If you change
anything here you have invalidated every measurement taken before the change.

Crafter uses the OLD Gym 4-tuple API, verified at crafter/env.py:118:

    obs, reward, done, info = env.step(action)

NOT the Gymnasium 5-tuple (obs, reward, terminated, truncated, info). We keep
the 4-tuple deliberately: `done` conflates death with hitting the step cap, and
the only thing that separates them is info['discount']. See THE EPISODE-END TRAP
below.
"""

import numpy as np
import crafter

# Verified against crafter/env.py.
OBS_SHAPE = (64, 64, 3)   # uint8, env.py:59
N_ACTIONS = 17            # env.py:63
EPISODE_CAP = 10000       # env.py:29


def make_env(seed, logdir=None, length=EPISODE_CAP):
    """Crafter plus the repo's own stats logger. No Gym wrappers.

    Args:
      seed: world generation + enemy AI seed.
      logdir: where stats.jsonl goes. None means no recording at all.
      length: episode step cap. Keep at 10000 for anything that gets reported.
    """
    env = crafter.Env(area=(64, 64), view=(9, 9), size=(64, 64),
                      reward=True, length=length, seed=seed)
    if logdir is None:
        return env
    # save_stats=True   -> writes stats.jsonl, which analysis/read_metrics.py
    #                      already knows how to read.
    # save_video=False  -> CRITICAL. VideoRecorder keeps every frame of the
    #                      episode in a Python list and flushes only on done.
    #                      A 10,000-step episode at 512x512x3 is ~7.8 GB. It
    #                      will OOM the box. Never enable on a training run.
    # save_episode=False-> same buffering problem. Only demos need it, and
    #                      record_demos.py does its own buffering.
    return crafter.Recorder(env, str(logdir), save_stats=True,
                            save_video=False, save_episode=False)


def bootstrap_mask(done, info):
    """1.0 if the future still exists, 0.0 if the player actually died.

    THE EPISODE-END TRAP. `done` is True in two unrelated situations
    (crafter/env.py:106-107):

        dead = self._player.health <= 0
        over = self._length and self._step >= self._length   # the 10,000 cap
        done = dead or over

    They need opposite treatment in the learning target:

      - Died         -> there is genuinely no future. Do not bootstrap.
      - Hit the cap  -> the game was fine, we stopped watching. MUST bootstrap.

    Get it wrong and you teach the agent that surviving 10,000 steps is worth
    zero, which poisons every long episode in the buffer.

        bootstrap_mask(done, info)         # RIGHT: 0.0 only on actual death
        0.0 if done else 1.0               # WRONG: also zeroes the timeout

    `done` is accepted only so that callers read naturally; the answer comes
    from info['discount'] alone, which crafter/env.py:111 sets to 0.0 on death.
    """
    del done
    return float(info['discount'])


class VecEnv:
    """Synchronous vector of Crafter envs with auto-reset.

    Moved here from baseline_agent/train.py so the DQN stages and the baseline
    share one implementation. Two differences from that copy: make_env is
    called by keyword (common.env.make_env takes (seed, logdir), the
    baseline_agent one took (outdir, seed) -- positionally these are swapped,
    which sets the seed to a path string), and num_envs=1 is a supported case
    so stages 02-10 keep their single-env behaviour by default.

    Every env gets its OWN Recorder writing to <logdir>/shardN/stats.jsonl.
    Two reasons this layout and not a single shared file: appending from 16
    envs to one handle interleaves partial lines, and analysis/read_metrics.py
    globs '**/stats.jsonl' treating each file as one run -- which is also how
    the random-agent runs already in logdir/ are laid out.

    Crafter's step() returns the old-gym 4-tuple, so this does too.
    """

    def __init__(self, num_envs, logdir=None, seed=0, length=EPISODE_CAP,
                 seed_stride=1):
        # SEED OVERLAP BETWEEN "INDEPENDENT" SEEDS. With seed_stride=1, env i
        # gets seed+i, so --seed 1 runs worlds 1..16 and --seed 2 runs worlds
        # 2..17: fifteen of sixteen world-generation streams are SHARED
        # between the two runs we report as two seeds. The policies still
        # differ (torch/numpy are seeded separately) and the streams drift as
        # episode counts diverge, but the seeds are less independent than the
        # label claims, which matters because the measured spread between two
        # seeds reaches 1.49 points.
        #
        # seed_stride=num_envs gives disjoint blocks. It is NOT the default:
        # it changes world generation, and common/env.py's own rule is that
        # changing this file invalidates every measurement taken before the
        # change. Pass --seed-stride 16 on new runs, keep 1 to reproduce old
        # ones.
        self.envs = [
            make_env(seed=seed * seed_stride + i,
                     logdir=f'{logdir}/shard{i}' if logdir else None,
                     length=length)
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
                # Auto-reset. The observation returned is the first of the next
                # episode; `done` tells the learner not to bootstrap across the
                # seam, via bootstrap_mask(done, info) on the info we return
                # here -- which is the dying step's info, not the new one's.
                #
                # THE TIMEOUT SEAM. On death that is enough, because mask is 0
                # and the frame is never bootstrapped from. On the 10,000-step
                # cap mask is 1.0 and SHOULD be: the game was fine, we stopped
                # watching. But then the learner bootstraps off the frame
                # below, which belongs to a freshly generated world. So keep
                # the real final observation where a caller can find it.
                # Nothing reads it yet by default -- 0 of 179,295 logged
                # episodes reached the cap, mean length ~180 -- so this is
                # recorded rather than acted on, and costs one dict entry.
                info = dict(info, final_obs=ob)
                ob = env.reset()
            # Facing is not in Crafter's info dict, but the extended action
            # mask needs it: `do` and the four moves all act on the one tile
            # the player faces. Read straight off the player -- an attribute
            # read, so it cannot change dynamics (Recorder forwards getattr).
            # Patching crafter/env.py instead would invalidate every
            # measurement taken before the change, per this file's own rule.
            info = dict(info, player_facing=tuple(env._player.facing))
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs), np.array(rewards, np.float32),
                np.array(dones), infos)
