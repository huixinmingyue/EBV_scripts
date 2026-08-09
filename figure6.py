"""Figure 6 | EBV+ CD8+ T cells in HLH carry an inflammatory, not a deeper
exhaustion, programme.

Core conclusion: within the two HLH CD8+ populations that matter for the
exhaustion trajectory - the exhausted subset and the state-1 cells at the head
of the trajectory - carrying EBV does not push cells further along the classical
exhaustion axis. Cytotoxic output is preserved and the separating signal is
inflammatory and interferon-related.

Panels
  a  Exhausted CD8+ T cells: programme-level module scores
  b  Exhausted CD8+ T cells: exhaustion genes
  c  Exhausted CD8+ T cells: inflammation genes
  d  Exhausted CD8+ T cells: cytotoxic genes
  e  State-1 CD8+ T cells: programme-level module scores
  f  State-1 CD8+ T cells: exhaustion genes
  g  State-1 CD8+ T cells: cytotoxic genes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import panels as P
from nature_style import (
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
    significance_stars,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig6"
OUT = ROOT / "Figure6"

STATUSES = ["EBV-", "EBV+"]
STATUS_COLORS = {"EBV-": "#9AA0A6", "EBV+": "#D55E00"}
STATUS_LABELS = {"EBV-": "EBV\u2212", "EBV+": "EBV+"}

MODULE_ORDER_EXH = ["Exhaustion", "Cytotoxic", "Inflammation", "Memory",
                    "Proliferation"]
MODULE_ORDER_S1 = ["Cell cycle", "DNA damage repair", "Interferon response",
                   "Exhaustion", "Cytotoxic"]

NOTES: list[str] = []


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order] * len(vals) / (np.arange(len(vals)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def compare(df, cat_col, value_col, order):
    """Mann-Whitney per category with Benjamini-Hochberg control inside a panel."""
    raw = []
    for cat in order:
        a = df.loc[(df[cat_col] == cat) & (df["EBV_status"] == "EBV-"),
                   value_col].dropna()
        b = df.loc[(df[cat_col] == cat) & (df["EBV_status"] == "EBV+"),
                   value_col].dropna()
        if len(a) < 3 or len(b) < 3:
            raw.append(np.nan)
        else:
            raw.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    adj = bh_adjust(raw)
    return {cat: (significance_stars(p) if np.isfinite(p) else "")
            for cat, p in zip(order, adj)}, adj


def gene_panel(ax, expr, genes, title, ylabel="Normalized expression"):
    stars, adj = compare(expr, "gene", "expression", genes)
    P.split_violin(ax, expr, "gene", "expression", "EBV_status", genes, STATUSES,
                   STATUS_COLORS, ylabel=ylabel, stars=stars)
    ax.set_title(title, fontsize=5.4, pad=2.0)
    return int(np.nansum(adj < 0.05))


def module_panel(ax, scores, order, title):
    stars, adj = compare(scores, "module", "score", order)
    P.split_violin(ax, scores, "module", "score", "EBV_status", order, STATUSES,
                   STATUS_COLORS, ylabel="Mean expression", italic=False,
                   stars=stars)
    ax.set_title(title, fontsize=5.4, pad=2.0)
    return int(np.nansum(adj < 0.05))


def main() -> None:
    apply_publication_style()

    exh_mod = pd.read_csv(DATA / "fig6_exhausted_modules.csv")
    exh_expr = pd.read_csv(DATA / "fig6_exhausted_expression.csv")
    exh_sets = pd.read_csv(DATA / "fig6_gene_sets_exhausted.csv")
    s1_mod = pd.read_csv(DATA / "fig6_state1_modules.csv")
    s1_expr = pd.read_csv(DATA / "fig6_state1_expression.csv")
    s1_sets = pd.read_csv(DATA / "fig6_gene_sets_state1.csv")

    def members(table, name):
        return table.loc[table["set"] == name, "gene"].tolist()

    exh_exhaustion = members(exh_sets, "exhaustion")
    exh_inflammation = members(exh_sets, "inflammation")
    exh_cytotoxic = members(exh_sets, "cytotoxic")
    s1_exhaustion = members(s1_sets, "Exhaustion")
    s1_cytotoxic = members(s1_sets, "Cytotoxic")

    n_exh = exh_mod.groupby("EBV_status")["cell_id"].nunique()
    n_s1 = s1_mod.groupby("EBV_status")["cell_id"].nunique()
    NOTES.append(f"panels a-d: {n_exh.sum():,} exhausted CD8+ T cells from HLH, "
                 f"{n_exh.get('EBV+', 0)} EBV+ and {n_exh.get('EBV-', 0)} EBV-")
    NOTES.append(f"panels e-g: {n_s1.sum():,} monocle2 state-1 CD8+ T cells from "
                 f"HLH, {n_s1.get('EBV+', 0)} EBV+ and {n_s1.get('EBV-', 0)} EBV-")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 6.35))
    outer = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1],
                             left=0.072, right=0.988, top=0.955, bottom=0.075,
                             hspace=0.72)

    row1 = outer[0].subgridspec(1, 2, width_ratios=[5, 6], wspace=0.20)
    ax_a = fig.add_subplot(row1[0])
    n_a = module_panel(ax_a, exh_mod, MODULE_ORDER_EXH, "Exhausted CD8+: modules")
    ax_b = fig.add_subplot(row1[1])
    n_b = gene_panel(ax_b, exh_expr, exh_exhaustion, "Exhausted CD8+: exhaustion")

    row2 = outer[1].subgridspec(1, 2, width_ratios=[4, 7], wspace=0.20)
    ax_c = fig.add_subplot(row2[0])
    n_c = gene_panel(ax_c, exh_expr, exh_inflammation,
                     "Exhausted CD8+: inflammation")
    ax_d = fig.add_subplot(row2[1])
    n_d = gene_panel(ax_d, exh_expr, exh_cytotoxic, "Exhausted CD8+: cytotoxic")

    row3 = outer[2].subgridspec(1, 2, width_ratios=[5, 10], wspace=0.20)
    ax_e = fig.add_subplot(row3[0])
    n_e = module_panel(ax_e, s1_mod, MODULE_ORDER_S1, "State-1 CD8+: modules")
    ax_f = fig.add_subplot(row3[1])
    n_f = gene_panel(ax_f, s1_expr, s1_exhaustion, "State-1 CD8+: exhaustion")

    ax_g = fig.add_subplot(outer[3])
    n_g = gene_panel(ax_g, s1_expr, s1_cytotoxic, "State-1 CD8+: cytotoxic")

    handles = [Line2D([], [], marker="s", linestyle="none", markersize=3,
                      markerfacecolor=STATUS_COLORS[s], markeredgecolor="none",
                      label=STATUS_LABELS[s]) for s in STATUSES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.988, 0.998),
               ncol=2, fontsize=5.5, handletextpad=0.4, columnspacing=1.0)

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                    (ax_e, "e"), (ax_f, "f"), (ax_g, "g")):
        panel_label(fig, ax, lab)

    NOTES.append(
        "every panel: left half-violin EBV\u2212, right half-violin EBV+, with the "
        "interquartile range and median marked. Two-sided Mann-Whitney per "
        "gene or module, Benjamini-Hochberg adjusted within the panel; "
        "*** q<0.001, ** q<0.01, * q<0.05.")
    NOTES.append(
        "categories within a panel share one y axis. The source figure scaled "
        "each gene independently, which made a gene expressed at 0.1 look like "
        "one expressed at 5.")
    NOTES.append(
        f"significant categories per panel: a {n_a}/{len(MODULE_ORDER_EXH)}, "
        f"b {n_b}/{len(exh_exhaustion)}, c {n_c}/{len(exh_inflammation)}, "
        f"d {n_d}/{len(exh_cytotoxic)}, e {n_e}/{len(MODULE_ORDER_S1)}, "
        f"f {n_f}/{len(s1_exhaustion)}, g {n_g}/{len(s1_cytotoxic)}")
    NOTES.append(
        "the source figure reserved an eighth, empty panel h; it is dropped "
        "because it carried no content.")

    written = save_figure(fig, OUT, "Figure6")
    (OUT / "Figure6_provenance.txt").write_text(
        "Figure 6 | EBV+ CD8+ T cells in HLH carry an inflammatory, not a deeper "
        "exhaustion, programme\nRendered in Python (matplotlib); R used only to "
        "export per-cell values from the stored analysis objects.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
