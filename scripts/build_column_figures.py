#!/usr/bin/env python3
"""Re-layout retained results at the CAS column width; no simulations."""
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_report import read, curve, OUT

DEST = Path(__file__).resolve().parents[1] / 'results/issue-1/figures/manuscript'
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.1, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    'pdf.fonttype': 42,
})

def canvas(rows=1, height=None):
    return plt.subplots(rows, 1, figsize=(3.35, height or 2.35 * rows),
                        layout='constrained', squeeze=False)

def save(fig, name):
    fig.savefig(DEST / f'sai_rev_{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

random = read('canonical_random.csv')
profiles = ['direct-sensitive', 'strict', 'moderate', 'tolerant']
for family, xlabel in [('ER', 'Edge probability $p$'),
                       ('BA', 'Attachment edges $m$'),
                       ('WS', r'Rewiring probability $\beta$')]:
    fig, axes = canvas(); ax = axes[0, 0]
    for profile in profiles:
        curve(ax, [r for r in random if r['family'] == family and r['profile'] == profile],
              'parameter', 'sai', profile)
    ax.set(xlabel=xlabel, ylabel='SAI (all ordered pairs)')
    ax.legend()
    if family == 'ER':
        for threshold in [1/50, np.log(50)/50]:
            ax.axvline(threshold, color='grey', lw=.6, ls=':', zorder=0)
    save(fig, family)

er = [r for r in random if r['family'] == 'ER' and r['profile'] == 'moderate']
fig, axes = canvas(2, 3.9); axes = axes[:, 0]
for key, label in [('sai', 'Total SAI'), ('support_term', 'Unreachability term'),
                   ('within_term', 'Within-support term')]:
    curve(axes[0], er, 'parameter', key, label)
for key, label in [('reachability', 'Reachable fraction'), ('mean', 'Mean, all pairs'),
                   ('mean_reachable', 'Mean, reachable pairs')]:
    curve(axes[1], er, 'parameter', key, label)
for ax in axes:
    ax.set_xlabel('Edge probability $p$'); ax.legend()
axes[0].set_ylabel('Normalized imbalance')
axes[1].set_ylabel('Fraction / satisfaction')
save(fig, 'coverage')

sens = read('sensitivity.csv')
aa = sorted({r['a'] for r in sens}); hh = sorted({r['h0'] for r in sens})
fig, axes = canvas(2, 3.8)
for ax, key, title in zip(axes[:, 0], ['sai', 'mean'], ['SAI', 'Mean satisfaction']):
    zz = np.array([[np.mean([r[key] for r in sens if r['a'] == a and r['h0'] == h])
                    for h in hh] for a in aa])
    im = ax.imshow(zz, origin='lower', aspect='auto', extent=[hh[0], hh[-1], 0, len(aa)])
    ax.set_yticks(np.arange(len(aa)) + .5, [f'{a:g}' for a in aa])
    ax.set(xlabel='$h_0$ (hops)', ylabel='$a$ (per hop)', title=title)
    fig.colorbar(im, ax=ax, pad=.02)
save(fig, 'sensitivity')

cal = read('calibration.csv'); fig, axes = canvas(2, 3.8)
for c1, c2 in sorted({(r['c1_nominal'], r['c2_nominal']) for r in cal}):
    rows = [r for r in cal if r['c1_nominal'] == c1 and r['c2_nominal'] == c2]
    for ax, key, label in zip(axes[:, 0], ['sai', 'mean'], ['SAI', 'Mean satisfaction']):
        ax.scatter([r['h0'] for r in rows], [r[key] for r in rows], s=12,
                   label=f'Anchors {c1:g}, {c2:g} hops')
        ax.set(xlabel='Calibrated $h_0$ (hops)', ylabel=label)
axes[0, 0].legend()
save(fig, 'calibration')

initial = read('dumbbell_initial.csv'); selected = read('edge_selected.csv')
edges = read('edge_candidates.csv')
topology = 'heterogeneous_15_30_path5'
before = next(r for r in initial if r['topology'] == topology and r['profile'] == 'tradeoff')
rows = [r for r in edges if r['topology'] == topology and r['profile'] == 'tradeoff']
fig, axes = canvas(); ax = axes[0, 0]
ax.scatter([before['mean_distance_reachable'] - r['mean_distance_reachable'] for r in rows],
           [r['sai'] - before['sai'] for r in rows], s=10, alpha=.5)
for r in selected:
    if r['topology'] == topology and r['objective'] in ['SAI', 'mean_distance']:
        ax.scatter(before['mean_distance_reachable'] - r['mean_distance_reachable'],
                   r['sai'] - before['sai'], s=55,
                   marker='x' if r['objective'] == 'SAI' else '+',
                   label='SAI' if r['objective'] == 'SAI' else 'Mean distance')
ax.axhline(0, c='grey', lw=.6)
ax.set(xlabel='Average distance reduction', ylabel='SAI change'); ax.legend()
save(fig, 'heterogeneous')

rows = [r for r in read('stress_edges.csv')
        if r['graph_id'] == 'modules4_regular4_2' and r['profile'] == 'tradeoff']
fig, axes = canvas(); ax = axes[0, 0]
ax.scatter([-r['delta_distance'] for r in rows], [r['delta_sai'] for r in rows], s=6, alpha=.35)
for edge, marker, color, label in [((1, 43), '*', '#cf382c', 'SAI optimum: (1, 43)'),
                                  ((5, 38), 'X', '#008c78', 'Distance optimum: (5, 38)')]:
    r = next(r for r in rows if (int(r['u']), int(r['v'])) == edge)
    ax.scatter(-r['delta_distance'], r['delta_sai'], s=75, marker=marker,
               color=color, label=label, zorder=5)
ax.set(xlabel='Average shortest-path reduction', ylabel='SAI change'); ax.legend()
save(fig, 'decision')

weighted = read('weighted.csv'); fig, axes = canvas(); ax = axes[0, 0]
for mode, label in [('hops', 'Hop costs: $a=1$, $h_0=5$'),
                     ('synthetic-cost', 'Synthetic costs: $a=0.8$, $h_0=12$')]:
    curve(ax, [r for r in weighted if r['cost_mode'] == mode], 'parameter', 'sai', label)
ax.set(xlabel='ER edge probability', ylabel='SAI'); ax.legend()
save(fig, 'weighted')

g = nx.read_edgelist(OUT / 'artifacts/two_cliques_25.edgelist', nodetype=int)
pos = {i: (np.cos(2*np.pi*i/25)-1.4, np.sin(2*np.pi*i/25)) for i in range(25)}
pos.update({i: (np.cos(2*np.pi*(i-25)/25)+1.4, np.sin(2*np.pi*(i-25)/25)) for i in range(25, 50)})
best = next(r for r in selected if r['topology'] == 'two_cliques_25' and r['objective'] == 'SAI')
fig, axes = canvas(2, 3.1)
for ax, title in zip(axes[:, 0], ['Initial dumbbell', 'After the selected edge is added']):
    nx.draw_networkx(g, pos, ax=ax, node_size=9, with_labels=False, width=.3, edge_color='#bbbbbb')
    ax.set_title(title); ax.axis('off')
nx.draw_networkx_edges(g, pos, edgelist=[(int(best['u']), int(best['v']))],
                       edge_color='#d33d28', width=1.4, ax=axes[1, 0])
save(fig, 'dumbbell')

data = read('runtime_memory.csv'); fig, axes = canvas(3, 4.8)
for backend in ['dense', 'batched']:
    rows = [r for r in data if r['backend'] == backend]
    for ax, key, label in zip(axes[:, 0],
                              ['path_and_histogram_seconds', 'aggregation_seconds', 'peak_rss_mib'],
                              ['Paths + histogram (s)', 'Aggregation (s)', 'Peak RSS (MiB)']):
        curve(ax, rows, 'n', key, backend)
        ax.set(xlabel='$N$', ylabel=label, xscale='log', yscale='log')
axes[0, 0].legend()
save(fig, 'runtime')

fig, axes = canvas(2, 3.9)
for family in sorted({r['family'] for r in random}):
    rows = [r for r in random if r['family'] == family and r['profile'] == 'zoo' and np.isfinite(r['sai'])]
    for ax, key, label in zip(axes[:, 0], ['degree_gini', 'mean'], ['Degree Gini', 'Mean satisfaction']):
        ax.scatter([r[key] for r in rows], [r['sai'] for r in rows], s=8, alpha=.4, label=family)
        ax.set(xlabel=label, ylabel='SAI')
axes[0, 0].legend(ncol=2)
save(fig, 'structure')

fig, axes = canvas(1, 2.8); ax = axes[0, 0]; ax.axis('off')
labels = ['Static graph + cost units', 'All-pair shortest-path costs\nUnreachable = infinity',
          'Declared anchors / profile\n$w$ = logistic(cost)', 'Report jointly\n$I$, coverage, mean, tails']
for y, label in zip([.92, .70, .47, .24], labels):
    ax.text(.5, y, label, ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=.4', fc='#edf3fa', ec='#315d83'))
for y1, y2 in [(.855, .785), (.62, .55), (.39, .32)]:
    ax.annotate('', xy=(.5, y2), xytext=(.5, y1), arrowprops=dict(arrowstyle='->', lw=.8))
ax.text(.5, .02, '$W=0$: $I$ undefined.\n$W>0$: normalize shares, compute entropy,\nseparate the support contribution.',
        ha='center', va='center', fontsize=7)
ax.set_ylim(-.07, 1)
save(fig, 'workflow')
print('Re-laid out 13 figures from retained results for one-column placement; no simulation run.')
