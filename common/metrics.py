"""The 22 achievement names, the Crafter score, and stats.jsonl reading.

This module is the single source of truth for "what is our number". Every plot,
every slide, every ablation row goes through compute_scores() here, so that when
we write "ours 4.8% vs Rainbow 4.31%" both numbers came out of the same code.
"""

import json
import pathlib
import warnings

import numpy as np

# All 22, checked against crafter/data.yaml. Alphabetical, which is also the
# order analysis/common.py produces from sorted() over the achievement_ keys.
ACHIEVEMENTS = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
    'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
    'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
    'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
    'place_stone', 'place_table', 'wake_up',
]
assert len(ACHIEVEMENTS) == 22


def compute_scores(percents):
    """Crafter score: geometric mean of the 22 success rates, offset by 1%.

        score = exp(mean(log(1 + percent))) - 1

    Hafner 2022 eq. 1, same as analysis/common.py:54. `percents` are PERCENTS,
    in [0, 100], not fractions in [0, 1]. The last axis must be the 22 tasks.

    The offset of 1% is why BREADTH BEATS POLISH: moving a task from 0% to 1%
    multiplies the product by 2, while moving one from 50% to 51% multiplies it
    by 1.02. Unlocking a new achievement at all is worth more than getting
    better at one we already have.

    This is the fixed version of the shipped assert. analysis/print_scores.py
    crashes on NumPy >= 2 at common.py:49 because agents have different seed
    counts, missing cells are padded with NaN, and `0 <= NaN` is False. We do
    not edit the vendored file; the fix lives here.
    """
    percents = np.asarray(percents, dtype=np.float64)
    assert np.all((percents >= 0) | np.isnan(percents)), 'percent below 0'
    assert np.all((percents <= 100) | np.isnan(percents)), 'percent above 100'
    # np.nanmean over an all-NaN slice warns; that slice is a genuinely empty
    # seed (an agent with fewer seeds than another, padded with NaN), and NaN
    # is the right answer for it, so let it through quietly.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_log = np.nanmean(np.log(1 + percents), -1)
    return np.exp(mean_log) - 1


def success_percents(unlocked_per_episode):
    """Per-achievement success rate, in percent, over a list of episodes.

    Args:
      unlocked_per_episode: list of dicts {achievement_name: count}, one per
        finished episode, exactly as info['achievements'] gives them.

    Returns an array of shape (22,) aligned with ACHIEVEMENTS. "Success" is
    count >= 1 in that episode, matching analysis/common.py:38.
    """
    if not unlocked_per_episode:
        return np.full(len(ACHIEVEMENTS), np.nan)
    counts = np.array([[float(ep.get(name, 0) >= 1) for name in ACHIEVEMENTS]
                       for ep in unlocked_per_episode])
    return 100.0 * counts.mean(0)


def read_stats(path):
    """Read one stats.jsonl into a list of per-episode achievement dicts.

    Written by crafter.Recorder's StatsRecorder (crafter/recorder.py:57-67):
    one JSON object per finished episode, with keys 'length', 'reward', and
    'achievement_<name>'.
    """
    episodes = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        episodes.append({name: record.get(f'achievement_{name}', 0)
                         for name in ACHIEVEMENTS})
    return episodes


def score_run(path, budget=None):
    """Score one seed's stats.jsonl. Returns (score, percents, n_episodes).

    budget: if set, only count episodes that FINISHED within this many env
    steps, matching the 1M-step budget convention in analysis/common.py:35.
    """
    episodes = read_stats(path)
    if budget is not None:
        lengths = [json.loads(line)['length']
                   for line in pathlib.Path(path).read_text().splitlines()
                   if line.strip()]
        keep = np.cumsum(lengths) <= budget
        episodes = [ep for ep, k in zip(episodes, keep) if k]
    percents = success_percents(episodes)
    return float(compute_scores(percents)), percents, len(episodes)


def score_pooled(paths, budget=None):
    """Score several shards of ONE run as a single pool of episodes.

    A num_envs > 1 run writes <logdir>/shardN/stats.jsonl per env. Those
    shards are one experiment, not N experiments, so the episodes are pooled
    and scored once.

    Do not average per-shard scores instead -- that is a different number,
    and which way it is wrong depends on the episode count per shard. The
    score is a geometric mean over 22 achievement rates, so it is dominated by
    the rates near zero: a shard that never unlocked an achievement scores it
    at 0% and drags the whole mean down, while a shard that got one lucky
    unlock in few episodes reads it as a much higher rate than the run earned.
    Measured on an 8-env smoke run (2-3 episodes per shard), the per-shard
    mean came out at 1.26% against a pooled 1.89% -- the missing-achievement
    effect dominating. With more episodes per shard the bias runs the other
    way. Either way only the pooled number is the run's score.

    budget: as score_run, but applied per shard -- each env gets budget //
    len(paths) steps, since that is its share of the total.
    """
    paths = [pathlib.Path(p) for p in paths]
    per_shard = budget // len(paths) if budget is not None else None
    episodes = []
    for path in paths:
        if per_shard is None:
            episodes.extend(read_stats(path))
            continue
        shard = read_stats(path)
        lengths = [json.loads(line)['length']
                   for line in path.read_text().splitlines() if line.strip()]
        keep = np.cumsum(lengths) <= per_shard
        episodes.extend(ep for ep, k in zip(shard, keep) if k)
    percents = success_percents(episodes)
    return float(compute_scores(percents)), percents, len(episodes)
