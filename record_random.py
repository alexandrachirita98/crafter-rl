"""Record a video of the stage 01 random agent playing.

Same policy as stage_01_random/train.py -- uniform over the 17 actions, no
network, no mask. Plays a few episodes and writes the best-scoring one
(most achievements, then highest reward) as a video, so the deck can show
what 1.6% actually looks like next to the trained agent.

    python record_random.py                                   # -> media/random.mp4
    python record_random.py --out media/random.gif
    python record_random.py --episodes 8 --max-steps 600 --fps 10

Eval worlds use the same seed block as record_video.py (10_000 + seed), so the
clip is never a training world and lines up with the agent video.

.mp4 needs `pip install imageio-ffmpeg`. .gif needs only pillow.
"""

import argparse
import pathlib
import sys

import imageio
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from common.env import N_ACTIONS, make_env


def play_episode(env, rng, size, max_steps):
    """One random episode. -> (frames, reward, achievements)."""
    env.reset()
    frames = [env.render((size, size))]
    total, achievements = 0.0, {}
    for _ in range(max_steps):
        # The whole policy, copied from stage_01_random/train.py.
        action = int(rng.integers(N_ACTIONS))
        _, reward, done, info = env.step(action)
        total += float(reward)
        achievements = dict(info.get('achievements', achievements))
        if done:
            break
        frames.append(env.render((size, size)))
    return frames, total, achievements


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=pathlib.Path, default='media/random.mp4')
    p.add_argument('--episodes', type=int, default=4,
                   help='play this many, keep the best')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=2000,
                   help='frame cap per episode; 2000 at 512px is ~1.5 GB RAM')
    p.add_argument('--size', type=int, default=512, help='frame side, px')
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args(argv)

    best = None
    for ep in range(args.episodes):
        env = make_env(10_000 + args.seed + ep, logdir=None)
        rng = np.random.default_rng(args.seed + ep)
        frames, reward, ach = play_episode(
            env, rng, size=args.size, max_steps=args.max_steps)
        unlocked = sorted(k for k, v in ach.items() if v)
        print(f'episode {ep}: {len(frames)} steps  reward {reward:.1f}  '
              f'{len(unlocked)} achievements: {", ".join(unlocked) or "-"}')
        key = (len(unlocked), reward)
        if best is None or key > best[0]:
            best = (key, frames, ep, unlocked)

    _, frames, ep, unlocked = best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == '.gif':
        imageio.mimsave(args.out, frames, duration=1000 / args.fps, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps)
    print(f'\nwrote {args.out}  (episode {ep}, {len(frames)} frames, '
          f'{len(unlocked)} achievements)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
