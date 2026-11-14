"""Site figures. One fixed colour per method (validated categorical slots 1-3); text stays in ink colours.

Aqua sits below 3:1 contrast on white, so every figure is paired with a table in the notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import GROUPS, METHOD_LABELS, METHODS, PAPER

COLORS = {"nonprivate": "#2a78d6", "dpsgd": "#eb6834", "global_adapt": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _bars(ax, x, heights, errs, color, width, **kw):
    # White edges give the 2px gap between adjacent bars; SE whiskers in secondary ink.
    return ax.bar(x, heights, width, yerr=errs, color=color, edgecolor=SURFACE, linewidth=1.5,
                  error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2}, **kw)


def _label(ax, x, value, se):
    ax.annotate(f"{value:.1f}", (x, max(value, 0) + (se or 0)), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=INK)


def group_bars(summary, groups, ylabel, title=None, label_groups=(), group_names=None, ax=None):
    """Grouped bars: x = group, one bar per method; mean ± SE. `summary` comes from analysis.summarize."""
    methods = [m for m in METHODS if m in set(summary.method)]
    ax = ax or plt.subplots(figsize=(max(5, 0.9 * len(groups) + 2), 3.4))[1]
    s = summary.set_index(["method", "group"])
    width, x = 0.8 / len(methods), np.arange(len(groups))
    for i, m in enumerate(methods):
        mean = [s.loc[(m, g), "mean"] for g in groups]
        se = [s.loc[(m, g), "se"] for g in groups]
        bars = _bars(ax, x + (i - (len(methods) - 1) / 2) * width, mean, se, COLORS[m], width,
                     label=METHOD_LABELS[m])
        for bar, g, v, e in zip(bars, groups, mean, se):
            if g in label_groups:
                _label(ax, bar.get_x() + bar.get_width() / 2, v, e)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, [group_names.get(g, g) if group_names else g for g in groups])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0 if not title else 1.08), ncol=len(methods),
              handlelength=1, fontsize=9)
    return ax


def gap_bars(gaps, dataset, ax=None):
    """Privacy-cost gap per private method: ours (solid) vs Esipova et al. (hatched), if the paper has a target."""
    a, b = GROUPS[dataset]
    methods = list(gaps.index)
    ax = ax or plt.subplots(figsize=(4.6, 3.2))[1]
    x, width = np.arange(len(methods)), 0.38
    has_paper = dataset in PAPER
    shift = width / 2 if has_paper else 0
    for i, m in enumerate(methods):
        ours, ours_se = gaps.loc[m, "gap"], gaps.loc[m, "se"]
        _bars(ax, x[i] - shift, [ours], [ours_se], COLORS[m], width)
        _label(ax, x[i] - shift, ours, ours_se)
        if has_paper:
            paper, paper_se = PAPER[dataset][m]["gap"]
            ax.bar(x[i] + width / 2, paper, width, yerr=paper_se, facecolor=COLORS[m] + "40", edgecolor=COLORS[m],
                   hatch="///", linewidth=1, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
            _label(ax, x[i] + width / 2, paper, paper_se)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
    ax.set_ylabel(f"Privacy-cost gap  |π$_{{{a}}}$ − π$_{{{b}}}$|  (pp)")
    if has_paper:
        ax.legend(handles=[Patch(facecolor=INK_2, label="Ours"),
                       Patch(facecolor=INK_2 + "40", edgecolor=INK_2, hatch="///", label=PAPER[dataset]["source"])],
                  loc="upper right", fontsize=8, handlelength=1.4)
    return ax


def vs_n(tbl, ylabel, y="mean", x="n_train", xlabel="Training-set size n (log scale)", title=None, ax=None,
         figsize=(6, 3.4)):
    """Lines over training-set size n (or another sweep column `x`; log x), mean ± SE: one colour per method,
    and a line style per group when `tbl` has a `group` column. `tbl` is one of analysis.size_sweep's tables."""
    ax = ax or plt.subplots(figsize=figsize)[1]
    groups = list(dict.fromkeys(tbl["group"])) if "group" in tbl else [None]
    styles = dict(zip(groups, [("-", "o"), ("--", "s"), (":", "^")]))
    for m in [m for m in METHODS if m in set(tbl.method)]:
        for g in groups:
            t = tbl[(tbl.method == m) & ((tbl.group == g) if g is not None else True)].sort_values(x)
            ls, marker = styles[g]
            label = METHOD_LABELS[m] + (f" · {g}" if g is not None else "")
            ax.errorbar(t[x], t[y], yerr=t.se, color=COLORS[m], linestyle=ls, marker=marker, markersize=4,
                        linewidth=1.5, elinewidth=1, ecolor=INK_2, capsize=2, label=label)
    ns = sorted(set(tbl[x]))
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n / 1000:.0f}k" if x == "n_train" else f"{n:g}" for n in ns])
    ax.minorticks_off()
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8, handlelength=2.2)
    return ax
