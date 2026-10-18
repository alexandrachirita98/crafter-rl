"""Compare run_kinds across stages from their summary_seed*.json files.

Built for the 1-env vs N-env question: the summaries already carry num_envs,
score_percent, episodes and the per-achievement rates, so no re-scoring of
stats.jsonl is needed.

    python compare_runs.py                              # every run_kind found
    python compare_runs.py --name 200k_v2 200k_v16      # just these two
    python compare_runs.py --name 200k_v2 200k_v16 --achievements
    python compare_runs.py --stage stage_03_double --name 200k_v2 200k_v16

Seeds of the same (stage, run_kind) are aggregated: mean +- spread over seeds,
which is the number to quote. A single seed prints its score with no spread --
two seeds disagreeing by more than they differ from another run_kind means the
comparison is not yet decidable, and the output says so.
"""

import argparse
import glob
import json
import pathlib
from collections import defaultdict

from common.metrics import ACHIEVEMENTS

def load(stage_filter=None, name_filter=None):
    runs = defaultdict(list)
    for path in sorted(glob.glob('stage_*/results/*/summary_seed*.json')):
        try:
            d = json.loads(pathlib.Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stage, kind = d.get('stage'), d.get('run_kind')
        if stage_filter and stage not in stage_filter:
            continue
        if name_filter and kind not in name_filter:
            continue
        runs[(stage, kind)].append(d)
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    """Half the range over seeds. With one seed there is no spread to report."""
    xs = [x for x in xs if x is not None]
    return (max(xs) - min(xs)) / 2 if len(xs) > 1 else None


def fmt(value, width=6, places=2):
    return f'{value:{width}.{places}f}' if value is not None else ' ' * (width - 1) + '-'


def table(runs):
    rows = []
    for (stage, kind), seeds in runs.items():
        scores = [s.get('score_percent') for s in seeds]
        envs = {s.get('num_envs', 1) for s in seeds}
        budgets = {s.get('env_steps') for s in seeds}
        rows.append({
            'stage': stage,
            'kind': kind,
            'envs': sorted(envs),
            'budget': sorted(budgets),
            'seeds': sorted(s.get('seed') for s in seeds),
            'score': mean(scores),
            'spread': spread(scores),
            'episodes': mean([s.get('episodes') for s in seeds]),
            'reward': mean([s.get('mean_episode_reward') for s in seeds]),
            'eps': mean([s.get('final_epsilon') for s in seeds]),
            'fps': mean([s.get('fps') for s in seeds]),
            'minutes': mean([s.get('wall_minutes') for s in seeds]),
        })
    rows.sort(key=lambda r: (r['stage'], -(r['score'] or 0)))

    head = (f"{'stage':<28} {'run_kind':<16} {'envs':>4} {'budget':>8} "
            f"{'score%':>7} {'+-':>5} {'eps':>5} {'rew':>5} {'episodes':>8} "
            f"{'fps':>5} {'min':>5}  seeds")
    print(head)
    print('-' * len(head))
    for r in rows:
        envs = ','.join(str(e) for e in r['envs'])
        budget = ','.join(f'{b // 1000}k' for b in r['budget'] if b)
        print(f"{r['stage']:<28} {r['kind']:<16} {envs:>4} {budget:>8} "
              f"{fmt(r['score'], 7)} {fmt(r['spread'], 5)} "
              f"{fmt(r['eps'], 5)} {fmt(r['reward'], 5)} "
              f"{fmt(r['episodes'], 8, 0)} {fmt(r['fps'], 5, 0)} "
              f"{fmt(r['minutes'], 5, 0)}  {','.join(map(str, r['seeds']))}")
    return rows


def head_to_head(rows, a, b):
    """Per stage, a vs b, with the seed spread kept in view."""
    by_stage = defaultdict(dict)
    for r in rows:
        by_stage[r['stage']][r['kind']] = r

    pairs = [(s, k[a], k[b]) for s, k in sorted(by_stage.items())
             if a in k and b in k]
    if not pairs:
        print(f'\nno stage has both {a!r} and {b!r}; nothing to compare')
        return

    print(f'\n{a}  ->  {b}')
    head = (f"{'stage':<28} {a[:11]:>11} {b[:11]:>11} {'delta':>7} "
            f"{'budget':>12}  verdict")
    print(head)
    print('-' * len(head))

    for stage, ra, rb in pairs:
        if ra['score'] is None or rb['score'] is None:
            continue
        delta = rb['score'] - ra['score']
        # A delta smaller than the spread it sits on is not a result. Two
        # seeds only bracket the variance, they do not measure it, so the
        # bar is the SUM of the two half-ranges, not the larger one: with
        # spreads of 0.009 and 0.0085 a delta of 0.0095 is noise, and the
        # stricter test would have called it a regression.
        noise = (ra['spread'] or 0) + (rb['spread'] or 0)
        if ra['budget'] != rb['budget']:
            verdict = f"BUDGETS DIFFER {ra['budget']} vs {rb['budget']}"
        elif noise == 0:
            verdict = 'single seed, no spread known'
        elif abs(delta) <= noise:
            verdict = f'within seed spread (+-{noise:.2f}), not decidable'
        else:
            verdict = 'better' if delta > 0 else 'WORSE'
        budget = ','.join(f'{x // 1000}k' for x in ra['budget'] if x)
        print(f"{stage:<28} {ra['score']:11.2f} {rb['score']:11.2f} "
              f"{delta:+7.2f} {budget:>12}  {verdict}")

    deltas = [rb['score'] - ra['score'] for _, ra, rb in pairs
              if ra['score'] is not None and rb['score'] is not None]
    if deltas:
        print(f"\nmean delta over {len(deltas)} stage(s): "
              f"{sum(deltas) / len(deltas):+.2f} points")


def achievements(runs, names):
    """Per-achievement rates side by side. Shows WHERE a change paid off."""
    cols = [(stage, kind) for (stage, kind) in runs if kind in names]
    if not cols:
        return
    stages = sorted({s for s, _ in cols})
    for stage in stages:
        kinds = [k for s, k in cols if s == stage]
        kinds = [k for k in names if k in kinds]
        if len(kinds) < 2:
            continue
        print(f'\n{stage}: per-achievement success %, '
              f'mean over seeds')
        head = f"{'achievement':<22}" + ''.join(f'{k[:12]:>13}' for k in kinds)
        print(head)
        print('-' * len(head))
        for ach in ACHIEVEMENTS:
            cells = []
            for kind in kinds:
                seeds = runs[(stage, kind)]
                cells.append(mean([s.get('success_percent', {}).get(ach)
                                   for s in seeds]))
            # Hide rows that are zero everywhere; they are the majority and
            # they bury the ones that moved.
            if all(c in (None, 0) for c in cells):
                continue
            print(f'{ach:<22}' + ''.join(
                f'{fmt(c, 12, 1)} ' for c in cells))
        # The wall the other doc identified: stone gates everything past wood.
        gate = [mean([s.get('success_percent', {}).get('collect_stone')
                      for s in runs[(stage, k)]]) for k in kinds]
        print(f"{'(collect_stone gate)':<22}" + ''.join(
            f'{fmt(g, 12, 2)} ' for g in gate))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', nargs='*', default=None,
                   help='run_kind(s) to include. Two names also print a '
                        'head-to-head table, in the order given')
    p.add_argument('--stage', nargs='*', default=None,
                   help='restrict to these stage dirs')
    p.add_argument('--achievements', action='store_true',
                   help='also print per-achievement rates side by side')
    args = p.parse_args()

    runs = load(args.stage, args.name)
    if not runs:
        print('no summary_seed*.json matched. '
              'Run from the repo root; try --name with a run_kind that exists.')
        return 1

    rows = table(runs)
    if args.name and len(args.name) == 2:
        head_to_head(rows, args.name[0], args.name[1])
    if args.achievements and args.name:
        achievements(runs, args.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
