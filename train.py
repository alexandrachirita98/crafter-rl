"""Submission entry point: the best agent found during the project.

Bare `python train.py` starts a 1M-step run of the configuration that won,
stage 18 -- Double DQN with legal-action masking (extended + skip-grass) and
3-step returns. Measured at 1M: 10.51% and 8.66% score over two seeds
(mean 9.59%), against 1.51% for the random agent on the same budget.

The agent, the buffer and the loop all live under stage_*/; this file only
pins the hyperparameters. That keeps one implementation of each piece rather
than a copy that can drift from the one the numbers were measured with.

    python train.py                      # the best run, seed 1, 1M steps
    python train.py --seed 3             # another seed
    python train.py --total-steps 200000 # the development budget
    python train.py --eval-every 50000   # add greedy-eval curves
    python train.py --test-stack         # self-tests, no training

Equivalent explicit command, for the record:
    python stage_18_extmask_nstep/train.py --total-steps 1000000 \
        --num-envs 16 --seed 1 --mask --mask-extended --mask-skip-grass \
        --n-step 3 --seed-stride 16 --eps-decay-steps 250000 \
        --warmup-steps 5000 --buffer-size 500000
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from stage_18_extmask_nstep.train import main as stage_18_main

# The winning configuration. These are defaults, not overrides: anything the
# user passes on the command line wins, because argparse sees it later.
#
# --mask* and --n-step are the two changes that beat the stage 03 base. The
# rest are the budget and the schedule the 1M runs used; they differ from
# stage 02's argparse defaults, which are the 200k development values.
BEST = [
    '--total-steps', '1000000',
    '--num-envs', '16',
    '--mask',                 # legal-action masking (uses info, see masking.py)
    '--mask-extended',        # + do/sleep/movement rules
    '--mask-skip-grass',      # + no `do` at grass
    '--n-step', '3',
    '--seed-stride', '16',    # disjoint worlds per seed, not 15/16 shared
    '--eps-decay-steps', '250000',
    '--warmup-steps', '5000',
    '--buffer-size', '500000',  # ~13 GB host RAM; see README
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--test-stack' in argv:
        return stage_18_main(argv)

    # A flag already given on the command line is not re-supplied from BEST,
    # so `--total-steps 200000` does not end up passed twice. --no-mask has
    # to suppress --mask, and the store_true mask flags take no value.
    given = {a.split('=', 1)[0] for a in argv if a.startswith('--')}
    if '--no-mask' in given:
        given |= {'--mask', '--mask-extended', '--mask-skip-grass'}

    defaults, i = [], 0
    while i < len(BEST):
        flag = BEST[i]
        takes_value = (i + 1 < len(BEST)
                       and not BEST[i + 1].startswith('--'))
        if flag not in given:
            defaults.append(flag)
            if takes_value:
                defaults.append(BEST[i + 1])
        i += 2 if takes_value else 1

    return stage_18_main(defaults + argv)


if __name__ == '__main__':
    sys.exit(main() or 0)
