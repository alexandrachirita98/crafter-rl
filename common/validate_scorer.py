"""Stage 01 gate: prove common/metrics.py reproduces the published numbers.

Reads the baseline JSON files shipped in scores/ and scores them with OUR
compute_scores(). The published values from Hafner 2022 Table 1 are hard-coded
below; if ours disagree by more than the tolerance, our scorer is wrong and
every later comparison would be meaningless.

Run:  python -m common.validate_scorer
"""

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from analysis import common as shipped          # noqa: E402  vendored loader
from common.metrics import ACHIEVEMENTS, compute_scores   # noqa: E402

# The repo's own scoreboard, README.md lines 128-150, which is the authoritative
# published value for each of these runs because these are the very JSON files
# those numbers were computed from. Hafner 2022 Table 1 agrees.
#
# Taken from README.md verbatim rather than from memory: the first version of
# this file had dreamerv2 at 13.7%, which is DreamerV3's number, and the
# mismatch looked like a scorer bug for a few minutes. Check the table, do not
# recall the table.
PUBLISHED = {
    'dreamerv2': 10.0,
    'ppo': 4.6,
    'rainbow': 4.3,
    'random': 1.6,
    'unsup_plan2explore': 2.1,
    'unsup_rnd': 2.0,
    'human': 50.5,
}
TOLERANCE = 0.1   # percentage points; the scoreboard rounds to one decimal


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    files = sorted((root / 'scores').glob('*.json'))
    runs = shipped.load_runs(files, budget=1e6, verbose=False)
    percents, methods, seeds, tasks = shipped.compute_success_rates(runs)

    # The vendored loader sorts the achievement_ keys, so its task order must
    # equal ours. If this trips, our ACHIEVEMENTS list is out of sync.
    assert [t.replace('achievement_', '') for t in tasks] == ACHIEVEMENTS, \
        'achievement order mismatch between metrics.py and analysis/common.py'

    # percents is (method, seed, task). Score each seed, then average over
    # seeds -- the same order of operations as analysis/print_scores.py.
    scores = compute_scores(percents)          # -> (method, seed)
    per_method = np.nanmean(scores, -1)        # -> (method,)

    print(f'{"method":<26} {"ours":>7} {"published":>10} {"delta":>7}')
    print('-' * 54)
    failures = []
    for name, score in zip(methods, per_method):
        key = name.split('-')[-1]
        want = PUBLISHED.get(key)
        if want is None:
            print(f'{name:<26} {score:>6.2f}% {"":>10} {"":>7}   (no published value)')
            continue
        delta = score - want
        flag = 'OK' if abs(delta) <= TOLERANCE else 'FAIL'
        if flag == 'FAIL':
            failures.append((name, score, want))
        print(f'{name:<26} {score:>6.2f}% {want:>9.1f}% {delta:>+6.2f}  {flag}')

    print()
    if failures:
        for name, got, want in failures:
            print(f'FAIL {name}: ours {got:.2f}% vs published {want:.1f}%')
        return 1
    print(f'All {len(per_method)} scored methods match the published table '
          f'within {TOLERANCE} points. The scorer is trustworthy.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
