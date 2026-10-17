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
