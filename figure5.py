"""Figure 5 | Cellular tropism of EBV across the disease continuum.

Core conclusion: EBV-bearing cells are rare everywhere but are not distributed
at random. B/plasma cells carry the virus in IM, whereas in MH-CD4 and HLH the
burden shifts into the CD8+ compartment and concentrates in the cytotoxic and
pre-exhausted subsets that sit at the start of the exhaustion trajectory; within
HLH, EBV+ CD8+ cells carry a more inflammatory and less memory-like programme
than their EBV- neighbours.

Panels
  a  Immune-wide embedding with EBV+ cells overlaid
  b  Per-sample immune-wide EBV+ rate by group
  c  IM: EBV+ rate by lineage (pooled) and per IM sample
  d  Per-sample EBV+ rate by lineage across all groups
  e  CD8+ embedding with EBV+ cells overlaid
  f  Plasma EBV DNA against CD8+ EBV+ rate
  g  Pooled CD8+ EBV+ rate by group
  h  Per-sample CD8+ EBV+ rate by subtype
  i  CD8+ EBV+ rate by subtype, MH-CD4 against HLH
  j  CD8+ EBV+ rate by pseudotime state, MH-CD4 against HLH
  k  Where HLH EBV+ CD8+ cells sit along the state ordering
  l  Module scores of HLH pre-exhausted CD8+ cells, EBV+ against EBV-
  m  Differential expression in the same contrast
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import panels as P
from nature_style import (
    CELLTYPE_COLORS,
    DIVERGING_CMAP,
    EXPR_CMAP,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    add_sig_bracket,
    apply_publication_style,
    embedding_axes,
    panel_label,
    save_figure,
    significance_stars,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig5"
OUT = ROOT / "Figure5"

EBV_COLOR = "#E7298A"

LINEAGE_SHORT = {
    "Bcells": "B", "CD4T": "CD4T", "CD8T": "CD8T", "Erythroid_cells": "Ery",
    "Mono/Mac": "Mono/Mac", "NK": "NK", "NKT": "NKT",
    "Neutrophil/Eosinophil": "Neu/Eos", "Plasma cells": "Plasma",
    "Platelets": "Plt", "cDC": "cDC", "gdT_cells": "\u03b3\u03b4T", "pDC": "pDC",
}

SUBTYPE_ORDER = [
    "CD8T_CCR7_TCF7_IL7R", "CD8T_TCF7_IL7R", "CD8T_KLRB1_CCR7_IL7R",
    "CD8T_GZMH", "CD8T_GZMH_NKG7", "CD8T_GZMK_TIGIT_HAVCR2",
    "CD8T_MKI67", "CD8T_MHC",
]
SUBTYPE_LABEL = {s: s.replace("CD8T_", "").replace("_", " ") for s in SUBTYPE_ORDER}
SUBTYPE_LABEL_WRAP = dict(SUBTYPE_LABEL, **{
    "CD8T_CCR7_TCF7_IL7R": "CCR7 TCF7\nIL7R",
    "CD8T_KLRB1_CCR7_IL7R": "KLRB1 CCR7\nIL7R",
    "CD8T_GZMK_TIGIT_HAVCR2": "GZMK TIGIT\nHAVCR2",
})
# Marker-token abbreviations for the two axes where the full subtype names would
# collide; the full names are carried by the embedding in panel e.
SUBTYPE_SHORT = {
    "CD8T_CCR7_TCF7_IL7R": "CCR7", "CD8T_TCF7_IL7R": "TCF7",
    "CD8T_KLRB1_CCR7_IL7R": "KLRB1", "CD8T_GZMH": "GZMH",
    "CD8T_GZMH_NKG7": "GZMH NKG7", "CD8T_GZMK_TIGIT_HAVCR2": "GZMK TIGIT",
    "CD8T_MKI67": "MKI67", "CD8T_MHC": "MHC",
}
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER, [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]))

IM_SAMPLES = ["IM_1", "IM_2", "IM_3", "IM_4", "IM_6", "IM_7", "IM_8"]
MODULE_LABEL = {
    "Memory_score": "Memory", "Exhaustion_score": "Exhaustion",
    "Cytotoxicity_score": "Cytotoxicity", "Inflammation_score": "Inflammation",
    "Proliferation_score": "Proliferation",
}

NOTES: list[str] = []


def sqrt_axis(ax, ticks):
    """Square-root y scale: EBV+ rates span three orders of magnitude but include
    exact zeros, so neither a linear nor a log axis shows both ends."""
    ax.set_yscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                         lambda v: np.square(v)))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def pairwise_brackets(ax, df, group_col, value_col, order, top_frac=1.02):
    """Draw brackets for the significant Mann-Whitney contrasts only."""
    found = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a = df.loc[df[group_col] == order[i], value_col].dropna()
            b = df.loc[df[group_col] == order[j], value_col].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            star = significance_stars(p)
            if star:
                found.append((i, j, star))
    if not found:
        return
    lo, hi = ax.get_ylim()
    step = (hi - lo) * 0.11
    base = max(df[value_col].max(), hi * 0.5) * top_frac
    for k, (i, j, star) in enumerate(found):
        add_sig_bracket(ax, i, j, base + k * step, star)
    ax.set_ylim(lo, base + len(found) * step + step * 0.6)


def panel_a(fig, ax, cells):
    P.umap_by_category(ax, cells, "celltype", CELLTYPE_COLORS, LINEAGE_SHORT,
                       size=0.35, min_dist_frac=0.40)
    pos = cells[cells["ebv"]]
    ax.scatter(pos["UMAP1"], pos["UMAP2"], s=3.0, color=EBV_COLOR, linewidths=0.15,
               edgecolors="white", zorder=5)
    ax.set_title(f"{len(cells):,} immune cells  |  {len(pos)} EBV+", fontsize=5.2,
                 color=NEUTRAL_MID, pad=1.5)


def panel_b(ax, sample_rate):
    means, sems, xs = [], [], np.arange(len(GROUP_ORDER))
    rng = np.random.default_rng(11)
    for i, g in enumerate(GROUP_ORDER):
        v = sample_rate.loc[sample_rate["group"] == g, "rate"].to_numpy()
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=2.6,
                   color=NEUTRAL_DARK, alpha=0.8, linewidths=0, zorder=3)
    ax.bar(xs, means, width=0.62, color=[GROUP_COLORS[g] for g in GROUP_ORDER],
           linewidth=0, alpha=0.9)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor=NEUTRAL_DARK,
                elinewidth=0.4, capsize=1.2, capthick=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("EBV+ rate (%)\nall immune cells", labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.set_xlim(-0.65, len(GROUP_ORDER) - 0.35)
    pairwise_brackets(ax, sample_rate, "group", "rate", GROUP_ORDER)


def panel_c(fig, ax_bar, ax_map, cax, im_rate):
    im = im_rate[im_rate["sample_id"].isin(IM_SAMPLES)]
    pooled = (im.groupby("celltype3")
              .agg(n=("n_cells", "sum"), pos=("n_ebv_positive", "sum")).reset_index())
    pooled["rate"] = pooled["pos"] / pooled["n"] * 100
    pooled = pooled.sort_values("rate")
    y = np.arange(len(pooled))
    ax_bar.barh(y, pooled["rate"], height=0.7, linewidth=0,
                color=[EBV_COLOR if r > 0 else "#D5D5D5" for r in pooled["rate"]])
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([LINEAGE_SHORT.get(c, c) for c in pooled["celltype3"]])
    ax_bar.set_ylim(-0.7, len(pooled) - 0.3)
    ax_bar.set_xlabel("EBV+ rate (%)", labelpad=1.5)
    ax_bar.set_xlim(0, pooled["rate"].max() * 1.40)
    ax_bar.tick_params(length=1.2, pad=1.0)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax_bar.set_axisbelow(True)
    for yi, (_, row) in zip(y, pooled.iterrows()):
        if row["rate"] > 0:
            ax_bar.text(row["rate"] + pooled["rate"].max() * 0.03, yi,
                        f"{row['pos']:.0f}/{row['n']:,.0f}", va="center", ha="left",
                        fontsize=5.0, color=NEUTRAL_DARK)
    ax_bar.set_title(f"IM, {len(IM_SAMPLES)} samples", fontsize=5.2, pad=2.0)

    hit = pooled[pooled["rate"] > 0]["celltype3"].tolist()[::-1]
    mat = (im.pivot_table(index="celltype3", columns="sample_id",
                          values="ebv_positive_rate", aggfunc="first")
           .reindex(index=hit, columns=IM_SAMPLES) * 100).fillna(0.0).to_numpy()
    # Rates run from 0.004% to 11%, so a linear ramp would leave every lineage
    # except plasma cells indistinguishable from an empty cell.
    norm = mcolors.FuncNorm((np.log1p, np.expm1), vmin=0, vmax=mat.max())
    im_plot = ax_map.imshow(mat, cmap=EXPR_CMAP, aspect="auto", norm=norm)
    ax_map.set_xticks(range(len(IM_SAMPLES)))
    ax_map.set_xticklabels([s.replace("IM_", "") for s in IM_SAMPLES])
    ax_map.set_yticks(range(len(hit)))
    ax_map.set_yticklabels([LINEAGE_SHORT.get(c, c) for c in hit])
    ax_map.set_xlabel("IM sample", labelpad=1.5)
    ax_map.tick_params(length=0, pad=1.0)
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                ax_map.text(j, i, f"{v:.2g}", ha="center", va="center",
                            fontsize=5.0,
                            color="white" if v > 3 else NEUTRAL_DARK)
    cb = fig.colorbar(im_plot, cax=cax, ticks=[0, 0.5, 2, 10])
    cb.ax.set_title("EBV+\nrate (%)", fontsize=5.0, pad=2.0)
    cb.ax.set_yticklabels(["0", "0.5", "2", "10"])
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    return pooled


def panel_d(ax, lineage):
    order = (lineage.groupby("celltype")["rate"].mean()
             .sort_values(ascending=False).index.tolist())
    data = [lineage.loc[lineage["celltype"] == c, "rate"].to_numpy() for c in order]
    bp = ax.boxplot(data, widths=0.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color="white", linewidth=0.6),
                    boxprops=dict(linewidth=0.3, edgecolor=NEUTRAL_DARK),
                    whiskerprops=dict(linewidth=0.3, color=NEUTRAL_DARK),
                    capprops=dict(linewidth=0.3, color=NEUTRAL_DARK))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CELLTYPE_COLORS.get(c, NEUTRAL_MID))
        patch.set_alpha(0.8)
    rng = np.random.default_rng(5)
    for i, c in enumerate(order, start=1):
        sub = lineage[lineage["celltype"] == c]
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(sub)), sub["rate"], s=1.8,
                   color=[GROUP_COLORS[g] for g in sub["group"]], alpha=0.9,
                   linewidths=0, zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([LINEAGE_SHORT.get(c, c) for c in order], rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("EBV+ rate (%)\nper sample", labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    sqrt_axis(ax, [0, 0.1, 0.5, 1, 2, 5, 10])
    ax.set_ylim(0, max(v.max() for v in data if len(v)) * 1.10)
    p = stats.kruskal(*[d for d in data if len(d) > 0]).pvalue
    ax.set_title(f"Kruskal\u2013Wallis p = {p:.1e}", fontsize=5.2, pad=2.0,
                 color=NEUTRAL_MID)
    return order, p


def panel_e(ax, cd8):
    P.umap_by_category(ax, cd8, "celltype", SUBTYPE_COLORS, SUBTYPE_LABEL_WRAP,
                       size=0.35, min_dist_frac=0.44)
    pos = cd8[cd8["ebv"]]
    ax.scatter(pos["UMAP1"], pos["UMAP2"], s=3.0, color=EBV_COLOR, linewidths=0.15,
               edgecolors="white", zorder=5)
    ax.set_title(f"{len(cd8):,} CD8+ T cells  |  {len(pos)} EBV+", fontsize=5.2,
                 color=NEUTRAL_MID, pad=1.5)


def panel_f(ax, clinical, rho, pval):
    d = clinical.dropna(subset=["ebv_dna"]).copy()
    d = d[d["group"].isin(GROUP_ORDER)]
    x = np.log10(d["ebv_dna"].to_numpy() + 1.0)
    y = d["rate"].to_numpy() * 100
    for g in GROUP_ORDER:
        m = (d["group"] == g).to_numpy()
        if m.any():
            ax.scatter(x[m], y[m], s=6, color=GROUP_COLORS[g], linewidths=0.2,
                       edgecolors="white", label=GROUP_LABELS[g], zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 60)
    fit = slope * grid + intercept
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2)) * np.sqrt(
        1 / len(x) + (grid - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    ax.plot(grid, fit, color=NEUTRAL_DARK, linewidth=0.6, zorder=2)
    ax.fill_between(grid, fit - 1.96 * se, fit + 1.96 * se, color="#D5D5D5",
                    alpha=0.55, linewidth=0, zorder=1)
    ax.set_xlabel("Plasma EBV DNA (log10 copies/mL + 1)", labelpad=1.5)
    ax.set_ylabel("CD8+ EBV+ rate (%)", labelpad=1.5)
    ax.set_title(f"Spearman \u03c1 = {rho:.2f}, p = {pval:.1e}, n = {len(d)}",
                 fontsize=5.2, pad=2.0, color=NEUTRAL_MID)
    ax.tick_params(length=1.2, pad=1.0)
    ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.3, borderpad=0.15,
              labelspacing=0.25, borderaxespad=0.2)


def panel_g(ax, by_group, fisher):
    d = by_group.set_index("Newgroup6").reindex(GROUP_ORDER)
    vals = d["rate"].to_numpy() * 100
    xs = np.arange(len(GROUP_ORDER))
    ax.bar(xs, vals, width=0.62, color=[GROUP_COLORS[g] for g in GROUP_ORDER],
           linewidth=0)
    for xi, v, npos, ntot in zip(xs, vals, d["n_ebv_pos"], d["n_total"]):
        ax.text(xi, v + max(vals) * 0.03, f"{v:.2f}%\n{npos:,}/{ntot:,}",
                ha="center", va="bottom", fontsize=5.0, color=NEUTRAL_DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("EBV+ rate (%)\nCD8+ T cells", labelpad=1.5)
    ax.set_xlim(-0.65, len(GROUP_ORDER) - 0.35)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.tick_params(length=1.2, pad=1.0)

    idx = {g: i for i, g in enumerate(GROUP_ORDER)}
    adjacent = [("HC", "IM"), ("IM", "MH_CD4"), ("MH_CD4", "HLH")]
    key = {(r["group1"], r["group2"]): r["padj"] for _, r in fisher.iterrows()}
    lo, hi = ax.get_ylim()
    base = max(vals) * 1.12
    step = (hi - lo) * 0.10
    for k, pair in enumerate(adjacent):
        p = key.get(pair, key.get(pair[::-1]))
        if p is None:
            continue
        add_sig_bracket(ax, idx[pair[0]], idx[pair[1]], base + k * step,
                        significance_stars(p) or "ns")
    ax.set_ylim(lo, base + len(adjacent) * step)


def panel_h(ax, sub, min_cells=20):
    # A subtype seen in only a handful of cells produces rates of 50% or more
    # from a single positive cell, which would set the axis for everything else.
    d = sub[sub["group"].isin(["MH_CD4", "HLH"]) & (sub["n"] >= min_cells)]
    order = (d.groupby("celltype")["rate"].mean()
             .sort_values(ascending=False).index.tolist())
    rng = np.random.default_rng(3)
    means, sems = [], []
    for i, c in enumerate(order):
        v = d.loc[d["celltype"] == c, "rate"].to_numpy()
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=1.8,
                   color=NEUTRAL_DARK, alpha=0.75, linewidths=0, zorder=3)
    ax.bar(np.arange(len(order)), means, width=0.62,
           color=[SUBTYPE_COLORS[c] for c in order], linewidth=0, alpha=0.9)
    ax.errorbar(np.arange(len(order)), means, yerr=sems, fmt="none",
                ecolor=NEUTRAL_DARK, elinewidth=0.4, capsize=1.2, capthick=0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SUBTYPE_SHORT[c] for c in order], rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("EBV+ rate (%)\nper sample, mean \u00b1 s.e.m.", labelpad=1.5)
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.tick_params(length=1.2, pad=1.0)
    sqrt_axis(ax, [0, 0.5, 1, 2, 4, 8])
    ax.set_ylim(0, max(d["rate"].max(), max(means)) * 1.12)
    return order, sorted(d["sample"].unique())


def panel_k(ax, state_dist):
    d = state_dist.copy()
    xs = np.arange(len(d))
    vals = d["frac"].to_numpy() * 100
    ax.bar(xs, vals, width=0.6, color=EBV_COLOR, linewidth=0, alpha=0.9)
    for xi, v, n in zip(xs, vals, d["n"]):
        ax.text(xi, v + max(vals) * 0.03, f"{v:.1f}%\n(n = {n})", ha="center",
                va="bottom", fontsize=5.0, color=NEUTRAL_DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels(d["MonocleState"].astype(int).astype(str))
    ax.set_xlabel("Pseudotime state", labelpad=1.5)
    ax.set_ylabel("% of HLH EBV+\nCD8+ cells", labelpad=1.5)
    ax.set_ylim(0, max(vals) * 1.32)
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.tick_params(length=1.2, pad=1.0)


def main() -> None:
    apply_publication_style()

    immune = pd.read_csv(DATA / "fig5_immune_umap.csv")
    cd8 = pd.read_csv(DATA / "fig5_cd8_umap.csv")
    sample_rate = pd.read_csv(DATA / "fig5_immune_sample_rate.csv")
    lineage = pd.read_csv(DATA / "fig5_immune_sample_lineage.csv")
    im_rate = pd.read_csv(DATA / "fig5_im_lineage_rate.csv")
    cd8_sub = pd.read_csv(DATA / "fig5_cd8_sample_subtype.csv")
    by_group = pd.read_csv(DATA / "EBV_rate_by_group.csv")
    fisher = pd.read_csv(DATA / "EBV_rate_fisher_by_group.csv")
    clinical = pd.read_csv(DATA / "EBV_rate_with_clinical.csv").rename(
        columns={"Newgroup6": "group"})
    spear = pd.read_csv(DATA / "EBV_DNA_vs_CD8_EBV_rate_spearman.csv").iloc[0]
    by_ct = pd.read_csv(DATA / "cd8_ebv_by_celltype.csv")
    by_state = pd.read_csv(DATA / "cd8_ebv_by_state.csv")
    state_dist = pd.read_csv(DATA / "HLH_EBV_pos_State_distribution.csv")
    modules = pd.read_csv(DATA / "HLH_pex_EBV_module_wilcox.csv")
    de = pd.read_csv(DATA / "HLH_pex_EBVpos_vs_neg_DE_all.csv")
    key_genes = pd.read_csv(DATA / "HLH_pex_key_genes_EBVpos_vs_neg.csv")

    NOTES.append(
        "panels a-d: immune-wide EBV calls come from the HV4 targeted assay, which "
        f"covers 23 of the 31 sequenced samples; {len(immune):,} cells with a call "
        f"are shown and {int(immune['ebv'].sum())} are EBV+. Samples without an "
        "assay result are excluded rather than being scored EBV-negative.")
    NOTES.append(f"panels e-j: {len(cd8):,} CD8+ T cells over {GROUP_ORDER} "
                 f"(IM_M and MH_CD56 excluded), {int(cd8['ebv'].sum())} EBV+")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.62))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.86, 1.30, 1.48, 1.38, 1.55],
                             left=0.078, right=0.962, top=0.972, bottom=0.046,
                             hspace=0.54)

    # ---- row 1: a | e -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 2, wspace=0.08)
    ax_a = fig.add_subplot(row1[0])
    panel_a(fig, ax_a, immune)
    ax_e = fig.add_subplot(row1[1])
    panel_e(ax_e, cd8)

    # ---- row 2: b | g | f | k ---------------------------------------------
    row2 = outer[1].subgridspec(1, 4, width_ratios=[0.86, 0.92, 1.22, 0.80],
                                wspace=0.52)
    ax_b = fig.add_subplot(row2[0])
    panel_b(ax_b, sample_rate)
    ax_g = fig.add_subplot(row2[1])
    panel_g(ax_g, by_group, fisher)
    ax_f = fig.add_subplot(row2[2])
    panel_f(ax_f, clinical, float(spear["rho"]), float(spear["p"]))
    ax_k = fig.add_subplot(row2[3])
    panel_k(ax_k, state_dist)
    NOTES.append("panel g: Fisher exact test on pooled cell counts between "
                 "neighbouring groups, Benjamini-Hochberg adjusted")
    NOTES.append(f"panel k: {int(state_dist['n'].sum())} EBV+ CD8+ cells in HLH "
                 "distributed over the monocle2 states; states with no EBV+ cell "
                 "are omitted")

    # ---- row 3: c (bar + per-sample map) | d -------------------------------
    row3 = outer[2].subgridspec(1, 5, width_ratios=[0.90, 0.62, 0.030, 0.16, 1.45],
                                wspace=0.34)
    ax_c1 = fig.add_subplot(row3[0])
    ax_c2 = fig.add_subplot(row3[1])
    cax_c = fig.add_subplot(row3[2])
    ax_d = fig.add_subplot(row3[4])
    pooled = panel_c(fig, ax_c1, ax_c2, cax_c, im_rate)
    order_d, p_d = panel_d(ax_d, lineage)
    NOTES.append("panel c: counts beside each bar are EBV+ cells over cells "
                 "assayed, pooled across the seven IM samples")
    NOTES.append(f"panel d: one point per sample, coloured by group; "
                 f"Kruskal-Wallis across lineages p = {p_d:.2e}")

    # ---- row 4: h | i | j -------------------------------------------------
    row4 = outer[3].subgridspec(1, 3, width_ratios=[1.0, 1.20, 0.68], wspace=0.42)
    ax_h = fig.add_subplot(row4[0])
    order_h, samples_h = panel_h(ax_h, cd8_sub)

    ax_i = fig.add_subplot(row4[1])
    by_ct = by_ct.assign(rate=by_ct["ebv_ratio"] * 100)
    P.grouped_bars(ax_i, by_ct, "celltype_new", "Newgroup6", "rate",
                   SUBTYPE_ORDER, ["MH_CD4", "HLH"], GROUP_COLORS, SUBTYPE_SHORT,
                   value_fmt=lambda v: f"{v:.1f}", ylabel="EBV+ rate (%)")
    ax_i.legend([Line2D([], [], marker="s", linestyle="none", markersize=3,
                        markerfacecolor=GROUP_COLORS[g], markeredgecolor="none",
                        label=GROUP_LABELS[g]) for g in ("MH_CD4", "HLH")],
                [GROUP_LABELS[g] for g in ("MH_CD4", "HLH")], loc="upper left",
                fontsize=5.0, handletextpad=0.3, borderpad=0.15, labelspacing=0.25)

    ax_j = fig.add_subplot(row4[2])
    by_state = by_state.assign(rate=by_state["ebv_ratio"] * 100)
    states = sorted(by_state["State"].unique())
    P.grouped_bars(ax_j, by_state, "State", "Newgroup6", "rate", states,
                   ["MH_CD4", "HLH"], GROUP_COLORS,
                   {s: str(int(s)) for s in states},
                   value_fmt=lambda v: f"{v:.1f}", ylabel="EBV+ rate (%)",
                   rotate=0)
    ax_j.set_xlabel("Pseudotime state", labelpad=1.5)
    NOTES.append(f"panel h: {len(samples_h)} MH-CD4 and HLH samples; subtypes "
                 "ordered by mean per-sample rate")
    NOTES.append("panels i and j: pooled cell counts within each group, so a "
                 "subtype with no EBV+ cell shows an empty slot")

    # ---- row 5: l | m -----------------------------------------------------
    row5 = outer[4].subgridspec(1, 3, width_ratios=[0.16, 0.95, 1.20], wspace=0.34)
    ax_l = fig.add_subplot(row5[1])
    mod = modules.dropna(subset=["p"]).copy()
    mod["label"] = mod["module"].map(MODULE_LABEL).fillna(mod["module"])
    mod["star"] = mod["padj"].map(lambda p: significance_stars(p) or "ns")
    mod = mod.sort_values("median_EBVpos")
    P.dumbbell(ax_l, mod, "label", "median_EBVneg", "median_EBVpos",
               ["#8C8C8C", EBV_COLOR], ["EBV\u2212", "EBV+"], star_col="star")
    ax_l.legend(loc="lower left", fontsize=5.0, handletextpad=0.2, borderpad=0.15,
                labelspacing=0.25, borderaxespad=0.2)
    ax_l.set_title("HLH pre-exhausted CD8+ T cells", fontsize=5.2, pad=2.0,
                   color=NEUTRAL_MID)
    NOTES.append(
        "panel l: medians of the per-cell module scores with a two-sided Wilcoxon "
        "rank-sum test, Benjamini-Hochberg adjusted. Four of the nine modules "
        "scored identically in every cell and returned no test statistic, so they "
        "are omitted. Medians are shown rather than violins because the per-cell "
        "scores are not part of the exported analysis output.")

    ax_m = fig.add_subplot(row5[2])
    # Label the exhaustion/inflammation panel genes that actually reach
    # significance, plus the strongest hits overall, so the labels do not pile
    # up along the y = 0 line.
    named = de[~de["gene"].str.startswith("ENSG")]
    labelled = key_genes.loc[key_genes["p_val_adj"] < 0.05, "gene"].tolist()
    labelled += named.nsmallest(8, "p_val_adj")["gene"].tolist()
    labelled = list(dict.fromkeys(labelled))
    n_up, n_down = P.volcano(ax_m, de, labelled)
    ax_m.set_title("HLH pre-exhausted CD8+ T cells", fontsize=5.2, pad=2.0,
                   color=NEUTRAL_MID)
    NOTES.append(f"panel m: {len(de):,} genes tested, {n_up} up and {n_down} down "
                 "in EBV+ cells at adjusted p < 0.05 and |log2FC| >= 0.5")

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c1, "c"), (ax_d, "d"),
                    (ax_e, "e"), (ax_f, "f"), (ax_g, "g"), (ax_h, "h"),
                    (ax_i, "i"), (ax_j, "j"), (ax_k, "k"), (ax_l, "l"),
                    (ax_m, "m")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure5")
    (OUT / "Figure5_provenance.txt").write_text(
        "Figure 5 | Cellular tropism of EBV across the disease continuum\n"
        "Rendered in Python (matplotlib); R used only to export data from the "
        "Seurat objects.\n\n" + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
