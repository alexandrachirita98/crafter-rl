"""Compare the demo-recorder variants on the same unseen seeds.

Runs each planner over an identical seed range and prints achievement rates
side by side, so differences come from the planner and not the worlds.

record_demos.py was tuned on seeds 3000-3019 and reported on 5000-5039, so
7000+ is clean for all three. Crafter's seed variance is large, so use 20+.

    python eval_demo_variants.py --seeds 7000 --n 40
"""
import argparse
import collections

import numpy as np

import crafter
import record_demos
from record_demos import ScriptedExpert, play_episode
from record_demos_coal_first import CoalFirstExpert
from record_demos_opportunistic import OpportunisticExpert

VARIANTS = [
    ('baseline', ScriptedExpert),
    ('A:coal-first', CoalFirstExpert),
    ('B:opportunistic', OpportunisticExpert),
]
# The ones that decide whether C3 can teach the iron branch. The rest are
# printed too, further down.
HEADLINE = ['make_stone_pickaxe', 'make_stone_sword', 'collect_coal',
            'collect_iron', 'make_iron_pickaxe', 'place_furnace', 'wake_up']


def run(expert_cls, seeds, length):
    counts = collections.Counter()
    lengths = []
    for seed in seeds:
        ach, steps, transitions = play_episode(seed, length, expert_cls)
        counts.update(k for k, v in ach.items() if v > 0)
        lengths.append(steps)
        del transitions          # free the frames before the next episode
    return counts, lengths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=7000, help='first seed')
    p.add_argument('--n', type=int, default=40, help='how many seeds')
    p.add_argument('--length', type=int, default=10000)
    p.add_argument('--only', nargs='*', default=None,
                   help='variant names to run (default: all)')
    args = p.parse_args()

    seeds = range(args.seeds, args.seeds + args.n)
    variants = [(n, c) for n, c in VARIANTS
                if args.only is None or n in args.only]

    results = {}
    for name, cls in variants:
        print(f'running {name} over seeds {args.seeds}-{args.seeds + args.n - 1}...',
              flush=True)
        results[name] = run(cls, seeds, args.length)

    names = [n for n, _ in variants]
    width = max(len(n) for n in names) + 2
    col = max(12, width)

    print(f'\n{args.n} unseen seeds, {args.seeds}-{args.seeds + args.n - 1}\n')
    header = f'{"metric":<24}' + ''.join(f'{n:>{col}}' for n in names)
    print(header)
    print('-' * len(header))

    row = f'{"mean length":<24}'
    for n in names:
        row += f'{np.mean(results[n][1]):>{col}.0f}'
    print(row)
    row = f'{"max length":<24}'
    for n in names:
        row += f'{max(results[n][1]):>{col}}'
    print(row)
    print()

    print('headline achievements (count / %d):' % args.n)
    for ach in HEADLINE:
        row = f'  {ach:<22}'
        for n in names:
            row += f'{results[n][0][ach]:>{col}}'
        print(row)

    print('\nall achievements:')
    for ach in crafter.constants.achievements:
        if ach in HEADLINE:
            continue
        row = f'  {ach:<22}'
        for n in names:
            row += f'{results[n][0][ach]:>{col}}'
        print(row)

    print('\nA variant is worth recording with only if collect_coal is '
          'non-zero. All three at\nzero means the demo set cannot teach the '
          'iron branch.')


if __name__ == '__main__':
    main()
