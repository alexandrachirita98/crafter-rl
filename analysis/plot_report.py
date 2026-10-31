"""Every figure the assignment asks for, from the runs in stage_*/results/.

    python analysis/plot_report.py              # all figures -> plots/report/
    python analysis/plot_report.py --budget 1e6

Produces, in order of the assignment's list:
  reward_curve      (c) mandatory: average episodic reward vs env steps,
                        mean over seeds with a min-max band, random included
  score_curve           Crafter score (geometric mean over 22 rates) vs steps
  success_rates     (e) per-achievement success rate, log scale, as in [9]
  final_scores      (d) final score per method with the random baseline
  loss_q            (g) training loss and mean Q, scraped from train_*.log
  ablation              the 1M ladder: what each change was worth

The upstream analysis/ scripts read the paper's own JSON dumps; these read our
results/ layout directly (shards_seed<N>/shard*.jsonl + train_seed<N>.log), so
nothing has to be converted first.

Colors are the data-viz reference palette's categorical slots 1-3, validated
for CVD separation; every series is also direct-labeled, so identity never
rests on color alone.
"""

import argparse
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Repo root FIRST: analysis/common.py (the paper's helpers) otherwise
# shadows the common/ package, since Python puts this script's own
# directory at the front of sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.metrics import ACHIEVEMENTS, compute_scores, success_percents

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Categorical slots 1-3 (blue / orange / aqua) + a muted ink for the baseline.
C_AGENT = '#2a78d6'
C_CONTROL = '#eb6834'
C_RANDOM = '#52514e'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#d8d7d2'

# The runs that go in the report. Order is the legend order.
#   label, directory, seeds, color
RUNS = [
    ('Ours (mask + 3-step)', 'stage_18_extmask_nstep/results/1m_skipgrass_n3',
     (1, 2), C_AGENT),
    ('Double DQN (control)', 'stage_14_extended_mask/results/1m_control',
     (1, 2), C_CONTROL),
]
RANDOM = ('Random', 'stage_01_random/results', (1,), C_RANDOM)

# The 1M ablation ladder, cheapest change first. Each entry is one arm.
ABLATION = [
    ('Double DQN', 'stage_03_double/results/1m_base', (1, 2)),
    ('+ mask', 'stage_03_double/results/1m_fix_mask', (1, 2)),
    ('+ n-step(3)', 'stage_04_nstep/results/1m_n3_mask', (1, 2)),
    ('+ ext. mask', 'stage_14_extended_mask/results/1m_extmask', (1, 2)),
    ('+ skip-grass', 'stage_14_extended_mask/results/1m_skipgrass',
     (1, 2, 3, 4)),
    ('+ both (ours)', 'stage_18_extmask_nstep/results/1m_skipgrass_n3',
     (1, 2)),
]


def _style():
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 140,
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.edgecolor': GRID,
        'axes.labelcolor': INK2,
        'axes.titlecolor': INK,
        'text.color': INK,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'lines.linewidth': 2.0,     # 2px lines, per the mark spec
    })


def _axes(ax):
    """Recessive grid, y only -- the x axis is already a step count."""
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ loading


def episode_rows(path):
    """One stats.jsonl -> list of dicts with length, reward, achievements.

    common.metrics.read_stats drops reward and length (it only scores), and
    both are needed here: reward IS the mandatory plot and length is what
    turns an episode index into an env-step position.
    """
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def seed_episodes(rundir, seed):
    """Pool one seed's shards into a single time-ordered episode list.

    A 16-env run writes one stats.jsonl per shard. Each shard is a separate
    env advancing in lockstep, so an episode that ENDS at cumulative shard
    step s sits at env step s * num_shards on the shared budget clock -- that
    scaling is what puts all shards on one x axis. Pooling (not averaging per
    shard) is also the only correct way to score: see common.metrics.
    """
    rundir = ROOT / rundir
    shard_dir = rundir / f'shards_seed{seed}'
    if shard_dir.is_dir():
        paths = sorted(shard_dir.glob('shard*.jsonl'),
                       key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))
    else:
        flat = rundir / f'stats_seed{seed}.jsonl'
        paths = [flat] if flat.exists() else []
    if not paths:
        return []

    n = len(paths)
    out = []
    for path in paths:
        step = 0
        for row in episode_rows(path):
            step += int(row.get('length', 0))
            row = dict(row)
            row['_step'] = step * n
            out.append(row)
    out.sort(key=lambda r: r['_step'])
    return out


def binned_curve(episodes, value, budget, bins=40):
    """Mean of `value` per equal-width step bin. Returns (centers, means)."""
    if not episodes:
        return np.array([]), np.array([])
    steps = np.array([e['_step'] for e in episodes], dtype=float)
    vals = np.array([float(e.get(value, 0)) for e in episodes])
    keep = steps <= budget
    steps, vals = steps[keep], vals[keep]
    edges = np.linspace(0, budget, bins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, bins - 1)
    centers, means = [], []
    for b in range(bins):
        sel = idx == b
        if sel.any():
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(vals[sel].mean())
    return np.array(centers), np.array(means)


def binned_score(episodes, budget, bins=20):
    """Crafter score over the episodes in each step bin.

    The score is a geometric mean over 22 rates, so it needs a POOL of
    episodes, not one episode -- hence coarser bins than the reward curve.
    """
    if not episodes:
        return np.array([]), np.array([])
    steps = np.array([e['_step'] for e in episodes], dtype=float)
    keep = steps <= budget
    steps = steps[keep]
    eps = [e for e, k in zip(episodes, keep) if k]
    edges = np.linspace(0, budget, bins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, bins - 1)
    centers, scores = [], []
    for b in range(bins):
        sel = [e for e, i in zip(eps, idx) if i == b]
        if len(sel) >= 10:
            unlocked = [{k: e.get(f'achievement_{k}', 0) for k in ACHIEVEMENTS}
                        for e in sel]
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            scores.append(float(compute_scores(success_percents(unlocked))))
    return np.array(centers), np.array(scores)


def final_stats(rundir, seeds, budget):
    """Per-seed (score, mean reward) at the budget, pooled over shards."""
    scores, rewards = [], []
    for seed in seeds:
        eps = [e for e in seed_episodes(rundir, seed) if e['_step'] <= budget]
        if not eps:
            continue
        unlocked = [{k: e.get(f'achievement_{k}', 0) for k in ACHIEVEMENTS}
                    for e in eps]
        scores.append(float(compute_scores(success_percents(unlocked))))
        rewards.append(float(np.mean([e.get('reward', 0) for e in eps])))
    return np.array(scores), np.array(rewards)


def final_percents(rundir, seeds, budget):
    """Per-achievement success rate, mean over seeds. Shape (22,)."""
    per_seed = []
    for seed in seeds:
        eps = [e for e in seed_episodes(rundir, seed) if e['_step'] <= budget]
        if not eps:
            continue
        unlocked = [{k: e.get(f'achievement_{k}', 0) for k in ACHIEVEMENTS}
                    for e in eps]
        per_seed.append(success_percents(unlocked))
    if not per_seed:
        return np.full(len(ACHIEVEMENTS), np.nan)
    return np.nanmean(np.stack(per_seed), 0)


LOG_RE = re.compile(
    r'\[\s*\d+\]\[\s*(\d+)\].*?loss=\s*([-\d.naN]+), meanQ=\s*([-\d.naN]+)')


def scrape_log(rundir, seed):
    """(steps, loss, meanQ) from train_seed<N>.log.

    Loss and Q were printed but never written to a structured file, so the
    log IS the record for requirement (g). One row per --log-every.
    """
    path = ROOT / rundir / f'train_seed{seed}.log'
    if not path.exists():
        return np.array([]), np.array([]), np.array([])
    steps, loss, q = [], [], []
    for line in path.read_text().splitlines():
        m = LOG_RE.search(line)
        if m:
            try:
                lv, qv = float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            steps.append(int(m.group(1)))
            loss.append(lv)
            q.append(qv)
    return np.array(steps), np.array(loss), np.array(q)


def seed_band(curves):
    """Mean across seeds plus a min-max band, on a shared x grid."""
    curves = [(x, y) for x, y in curves if len(x)]
    if not curves:
        return None
    grid = curves[0][0]
    for x, _ in curves[1:]:
        if len(x) < len(grid):
            grid = x
    ys = [np.interp(grid, x, y) for x, y in curves]
    ys = np.stack(ys)
    return grid, ys.mean(0), ys.min(0), ys.max(0)


def _label_end(ax, x, y, text, color):
    """Direct label at the end of a series -- the relief the contrast WARN
    on the aqua/green slots obligates, and it also lets the reader identify
    a series without matching it against the legend box."""
    if len(x):
        ax.annotate(text, (x[-1], y[-1]), textcoords='offset points',
                    xytext=(4, 0), va='center', fontsize=7, color=color,
                    fontweight='medium')


# ------------------------------------------------------------------ figures


def fig_reward(outdir, budget):
    """(c) MANDATORY: average episodic reward. Random included per (d)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _axes(ax)
    for label, rundir, seeds, color in RUNS + [RANDOM]:
        curves = [binned_curve(seed_episodes(rundir, s), 'reward', budget)
                  for s in seeds]
        band = seed_band(curves)
        if band is None:
            print(f'  !! no episodes for {rundir}')
            continue
        x, mean, lo, hi = band
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, label=f'{label} (n={len(seeds)})')
        _label_end(ax, x, mean, label.split(' (')[0], color)
    ax.set_xlabel('environment steps')
    ax.set_ylabel('average episodic reward')
    ax.set_title('Average episodic reward (mean over seeds, min-max band)')
    ax.set_xlim(0, budget)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(outdir / 'reward_curve.png', bbox_inches='tight')
    fig.savefig(outdir / 'reward_curve.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_score(outdir, budget):
    """Crafter score during training -- the headline metric of [9]."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _axes(ax)
    for label, rundir, seeds, color in RUNS + [RANDOM]:
        curves = [binned_score(seed_episodes(rundir, s), budget)
                  for s in seeds]
        band = seed_band(curves)
        if band is None:
            continue
        x, mean, lo, hi = band
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, label=f'{label} (n={len(seeds)})')
        _label_end(ax, x, mean, label.split(' (')[0], color)
    ax.set_xlabel('environment steps')
    ax.set_ylabel('Crafter score (%)')
    ax.set_title('Crafter score during training (geometric mean, 22 tasks)')
    ax.set_xlim(0, budget)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(outdir / 'score_curve.png', bbox_inches='tight')
    fig.savefig(outdir / 'score_curve.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_success(outdir, budget):
    """(e) Per-achievement success rate, as in [9] Fig. 4.

    Log x, because the rates span wake_up at ~95% down to diamond at 0%; a
    linear axis would flatten the whole tech tree into the baseline.
    """
    arms = [(lab, d, s, c) for lab, d, s, c in RUNS] + [RANDOM]
    data = {lab: final_percents(d, s, budget) for lab, d, s, _ in arms}
    order = np.argsort(-data[arms[0][0]])
    names = [ACHIEVEMENTS[i].replace('_', ' ') for i in order]
    y = np.arange(len(order))
    h = 0.8 / len(arms)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.grid(True, axis='x', zorder=0)
    ax.set_axisbelow(True)
    floor = 0.01
    for k, (label, _, seeds, color) in enumerate(arms):
        vals = np.clip(data[label][order], floor, None)
        ax.barh(y + (len(arms) - 1 - k) * h - 0.4 + h / 2, vals, height=h * 0.86,
                color=color, label=f'{label} (n={len(seeds)})', zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(floor, 150)
    ax.set_xlabel('success rate (%, log scale)')
    ax.set_title('Per-achievement success rate at 1M steps')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(outdir / 'success_rates.png', bbox_inches='tight')
    fig.savefig(outdir / 'success_rates.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_loss_q(outdir, budget):
    """(g) Training loss and mean Q, scraped from the logs."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax in axes:
        _axes(ax)
    for label, rundir, seeds, color in RUNS:
        lc, qc = [], []
        for s in seeds:
            st, ls, qs = scrape_log(rundir, s)
            if len(st):
                lc.append((st.astype(float), ls))
                qc.append((st.astype(float), qs))
        for ax, curves, name in ((axes[0], lc, 'loss'), (axes[1], qc, 'Q')):
            band = seed_band(curves)
            if band is None:
                continue
            x, mean, lo, hi = band
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, label=label, linewidth=1.6)
    axes[0].set_ylabel('TD loss (Huber)')
    axes[0].set_title('Training loss')
    axes[0].set_yscale('log')
    axes[1].set_ylabel('mean Q over batch')
    axes[1].set_title('Mean Q-value')
    for ax in axes:
        ax.set_xlabel('environment steps')
        ax.set_xlim(0, budget)
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(outdir / 'loss_q.png', bbox_inches='tight')
    fig.savefig(outdir / 'loss_q.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_ablation(outdir, budget):
    """What each change was worth at 1M. Per-seed dots over the mean bar."""
    labels, means, allseeds = [], [], []
    for label, rundir, seeds in ABLATION:
        sc, _ = final_stats(rundir, seeds, budget)
        if not len(sc):
            continue
        labels.append(f'{label}\n(n={len(sc)})')
        means.append(sc.mean())
        allseeds.append(sc)

    rnd, _ = final_stats(RANDOM[1], RANDOM[2], budget)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    _axes(ax)
    x = np.arange(len(labels))
    # Last bar is ours; the rest are intermediate arms.
    colors = [C_CONTROL] * (len(labels) - 1) + [C_AGENT]
    ax.bar(x, means, width=0.62, color=colors, zorder=2)
    for i, sc in enumerate(allseeds):
        ax.scatter([i] * len(sc), sc, s=14, color='#ffffff',
                   edgecolor=INK, linewidth=0.9, zorder=4)
    # Label above the highest seed dot, not the bar top: with n=2 a dot sits
    # exactly where a bar-top label would go and overwrites it.
    for i, (m, sc) in enumerate(zip(means, allseeds)):
        ax.annotate(f'{m:.1f}', (i, max(m, sc.max())),
                    textcoords='offset points', xytext=(0, 6), ha='center',
                    fontsize=7, color=INK, fontweight='medium')
    if len(rnd):
        ax.axhline(rnd.mean(), color=C_RANDOM, linestyle=':', linewidth=1.4,
                   zorder=3)
        # Outside the plot on the right: every bar clears the baseline, so
        # there is no in-axes gap for this label to sit in.
        ax.annotate(f'random {rnd.mean():.1f}%',
                    (len(labels) - 0.4, rnd.mean()),
                    textcoords='offset points', xytext=(6, -2), ha='left',
                    fontsize=7, color=C_RANDOM, annotation_clip=False)
    ax.set_ylim(0, max(max(s.max() for s in allseeds), 1) * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Crafter score (%)')
    ax.set_title('Ablation at 1M steps (bar = seed mean, dots = seeds)')
    fig.tight_layout()
    fig.savefig(outdir / 'ablation.png', bbox_inches='tight')
    fig.savefig(outdir / 'ablation.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_final(outdir, budget):
    """(d) Final score and reward per method, random included."""
    arms = list(RUNS) + [RANDOM]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax in axes:
        _axes(ax)
    for k, (metric, name) in enumerate((('score', 'Crafter score (%)'),
                                        ('reward', 'mean episodic reward'))):
        ax = axes[k]
        hi = []
        for i, (label, rundir, seeds, color) in enumerate(arms):
            sc, rw = final_stats(rundir, seeds, budget)
            vals = sc if metric == 'score' else rw
            if not len(vals):
                continue
            ax.bar(i, vals.mean(), width=0.6, color=color, zorder=2)
            ax.scatter([i] * len(vals), vals, s=16, color='#ffffff',
                       edgecolor=INK, linewidth=0.9, zorder=4)
            # Above the highest seed dot; a bar-top label collides with it.
            ax.annotate(f'{vals.mean():.2f}',
                        (i, max(vals.mean(), vals.max())),
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7, color=INK,
                        fontweight='medium')
            hi.append(vals.max())
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0].replace(' (', '\n(') for a in arms],
                           fontsize=6.5)
        ax.set_ylabel(name)
        if hi:
            ax.set_ylim(0, max(hi) * 1.16)
    axes[0].set_title('Final score at 1M')
    axes[1].set_title('Final average reward at 1M')
    fig.tight_layout()
    fig.savefig(outdir / 'final_scores.png', bbox_inches='tight')
    fig.savefig(outdir / 'final_scores.pdf', bbox_inches='tight')
    plt.close(fig)


def table(budget):
    """The numbers behind the figures, for the slides' text."""
    print('\nFinal results at '
          f'{budget:,.0f} steps  (score %, mean over seeds)')
    print(f'  {"arm":<26} {"score":>16}  {"reward":>14}  seeds')
    for label, rundir, seeds in (
            [(a, b, c) for a, b, c, _ in RUNS]
            + [(RANDOM[0], RANDOM[1], RANDOM[2])]
            + [a for a in ABLATION]):
        sc, rw = final_stats(rundir, seeds, budget)
        if not len(sc):
            print(f'  {label:<26} {"-- no data --":>16}')
            continue
        spread = f'{sc.mean():5.2f} ({sc.min():.2f}-{sc.max():.2f})'
        print(f'  {label:<26} {spread:>16}  {rw.mean():14.2f}  {len(sc)}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--budget', type=float, default=1e6,
                   help='env steps to score at; episodes after it are cut')
    p.add_argument('--outdir', type=str, default='plots/report')
    p.add_argument('--only', type=str, default=None,
                   help='one of reward, score, success, loss, ablation, final')
    args = p.parse_args(argv)

    budget = int(args.budget)
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    _style()

    figs = {
        'reward': fig_reward, 'score': fig_score, 'success': fig_success,
        'loss': fig_loss_q, 'ablation': fig_ablation, 'final': fig_final,
    }
    todo = [args.only] if args.only else list(figs)
    for name in todo:
        print(f'plotting {name} ...', flush=True)
        figs[name](outdir, budget)
    table(budget)
    print(f'\nwrote {len(todo)} figures (png + pdf) to {outdir}')


if __name__ == '__main__':
    main()
