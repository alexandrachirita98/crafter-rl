"""Record a video of the submission agent (stage 18) playing greedily.

Loads the weights checkpoint that `train.py --ckpt-every N` writes at
<logdir>/crafter_reward-stage_18_extmask_nstep/<seed>/ckpt.pt, plays a few
episodes with eps=0 and the same legal-action mask the run trained with, and
writes the best-scoring episode (most achievements, then highest reward) as
a video.

    python record_video.py --ckpt path/to/ckpt.pt                  # -> media/agent.gif
    python record_video.py --ckpt ckpt.pt --out media/agent.mp4    # needs imageio-ffmpeg
    python record_video.py --ckpt ckpt.pt --episodes 8 --fps 10
    python record_video.py --random --max-steps 200                # smoke test, no weights

Eval worlds use the same seed block as evaluate() in stage 02 (10_000 + seed),
so the recorded episode is never a training world.

.gif needs only pillow. .mp4 needs `pip install imageio-ffmpeg`.
"""

import argparse
import pathlib
import sys

import imageio
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from common.env import N_ACTIONS, VecEnv
from common.masking import legal_masks
from stage_02_dqn.train import _apply_mask, _to_tensor, get_estimator

# Must match the checkpoint's architecture. These are stage 02's argparse
# defaults, which every stage 18 run used.
LIN_SIZE = 64
STRIDE = 1


def load_net(ckpt):
    net = get_estimator(N_ACTIONS, lin_size=LIN_SIZE, stride=STRIDE)
    if ckpt is not None:
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        net.load_state_dict(ck['net'])
        print(f'loaded {ckpt}  (trained {ck["step"]:,} steps, '
              f'{ck["episodes"]} episodes)')
    net.to(_to_tensor(np.zeros((64, 64, 3), np.uint8)).device)
    return net.eval()


def play_episode(net, env, use_mask, extended, skip_grass, size, max_steps):
    """One greedy episode on env (a 1-env VecEnv). -> (frames, reward, achievements)."""
    state = env.reset()
    mask = None
    frames = [env.envs[0].render((size, size))]
    total, achievements = 0.0, {}
    for _ in range(max_steps):
        with torch.no_grad():
            q = net(_to_tensor(state))
            if mask is not None:
                q = _apply_mask(q, mask)
            action = q.argmax(1).cpu().numpy()
        state, rew, done, infos = env.step(action)
        total += float(rew[0])
        achievements = dict(infos[0].get('achievements', achievements))
        if done[0]:
            # VecEnv auto-reset: the last real frame is info['final_obs'],
            # but render() already shows the new world, so stop here.
            break
        frames.append(env.envs[0].render((size, size)))
        mask = (legal_masks(infos, done, extended=extended,
                            skip_grass=skip_grass) if use_mask else None)
    return frames, total, achievements


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--ckpt', type=pathlib.Path,
                   help='ckpt.pt from a --ckpt-every run')
    p.add_argument('--random', action='store_true',
                   help='untrained net, for checking the pipeline')
    p.add_argument('--out', type=pathlib.Path, default='media/agent.gif')
    p.add_argument('--episodes', type=int, default=4,
                   help='play this many, keep the best')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=2000,
                   help='frame cap per episode; 2000 at 512px is ~1.5 GB RAM')
    p.add_argument('--size', type=int, default=512, help='frame side, px')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--no-mask', action='store_true',
                   help='act without the legal-action mask (stage 18 used it)')
    args = p.parse_args(argv)

    if not args.ckpt and not args.random:
        p.error('give --ckpt path/to/ckpt.pt, or --random for a smoke test')

    net = load_net(None if args.random else args.ckpt)
    use_mask = not args.no_mask
    best = None
    for ep in range(args.episodes):
        env = VecEnv(1, logdir=None, seed=10_000 + args.seed + ep)
        frames, reward, ach = play_episode(
            net, env, use_mask, extended=True, skip_grass=True,
            size=args.size, max_steps=args.max_steps)
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
