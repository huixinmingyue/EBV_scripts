"""Figure 4 | Pseudotemporal ordering resolves CD8+ cell states along the continuum.

Core conclusion: a monocle2 trajectory rooted in the healthy-dominated state orders
CD8+ cells into five states whose occupancy shifts with disease severity, and the
branch that leads away from the homeostatic root carries an exhaustion programme.

Panels
  a  DDRTree trajectory coloured by state
  b  The same trajectory split by disease group
  c  Per-sample state occupancy across groups
  d  Which CD8+ subtype occupies which state
  e  Genes that fall and rise along pseudotime
  f  Subtype-by-state frequency per disease group
  g  State-defining markers
  h  KEGG enrichment of the two divergent states
  i  Branch-dependent genes at the first branch point
  j  Branched expression of those genes along pseudotime
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
    DIVERGING_CMAP,
    EXPR_CMAP,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    embedding_axes,
    panel_label,
    save_figure,
    significance_stars,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig4"
OUT = ROOT / "Figure4"
T04 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig04/TC_EBV_04_monocle2_reroot_HC")
XR = Path("F:/EBV/EBV_xiaomi/EBV/Rdata")
STATE_PREFIX = "cd8_monocle2_no_regress_state_specific"
BEAM_PREFIX = ("cd8_monocle2_BEAM_graph_test_branch_dependent_genes_analysis_"
               "branch_point_1_State4_vs_State3")

STATES = ["1", "2", "3", "4", "5"]
STATE_COLORS = dict(zip(STATES, ["#D1495B", "#66A182", "#2E4057", "#EDAE49", "#8D6A9F"]))
SUBTYPE_ORDER = [
    "CD8T_CCR7_TCF7_IL7R", "CD8T_TCF7_IL7R", "CD8T_KLRB1_CCR7_IL7R",
    "CD8T_GZMH", "CD8T_GZMH_NKG7", "CD8T_GZMK_TIGIT_HAVCR2",
    "CD8T_MKI67", "CD8T_MHC",
]
SUBTYPE_LABEL = {s: s.replace("CD8T_", "").replace("_", " ") for s in SUBTYPE_ORDER}
# Panel f pairs every subtype with a state, so its row names must be far shorter
# than the full marker strings used elsewhere.
SUBTYPE_SHORT = {
    "CD8T_CCR7_TCF7_IL7R": "CCR7/TCF7", "CD8T_TCF7_IL7R": "TCF7/IL7R",
    "CD8T_KLRB1_CCR7_IL7R": "KLRB1/CCR7", "CD8T_GZMH": "GZMH",
    "CD8T_GZMH_NKG7": "GZMH/NKG7", "CD8T_GZMK_TIGIT_HAVCR2": "GZMK/TIGIT",
    "CD8T_MKI67": "MKI67", "CD8T_MHC": "MHC",
}
HOME_COLOR = "#2C6FAD"
PROG_COLOR = "#B02418"
BRANCH_COLORS = {"Pre-branch": "#8C8C8C", "Fate1_State4": "#B02418",
                 "Fate2_State3": "#2C6FAD"}
BRANCH_LABEL = {"Pre-branch": "Pre-branch", "Fate1_State4": "State 4",
                "Fate2_State3": "State 3"}

NOTES: list[str] = []


def main() -> None:
    apply_publication_style()

    traj = pd.read_csv(DATA / "fig4_trajectory.csv", dtype={"State": str})
    traj = traj[traj["Newgroup6"].isin(GROUP_ORDER)].copy()
    NOTES.append(f"panels a,b,c,f: {len(traj):,} CD8+ cells on the DDRTree "
                 "trajectory rooted at the HC-dominated state; IM_M and MH_CD56 "
                 "cells present in the CDS are excluded to keep the four-group "
                 "continuum")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.72))
    outer = fig.add_gridspec(6, 1, height_ratios=[1.30, 1.05, 1.45, 2.15, 1.75, 1.10],
                             left=0.070, right=0.950, top=0.976, bottom=0.038,
                             hspace=0.42)
    letters: list = []

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 5, wspace=0.06)
    ax_a = fig.add_subplot(row1[0])
    for st in STATES:
        sub = traj[traj["State"] == st]
        ax_a.scatter(sub["Component1"], sub["Component2"], s=0.9, linewidths=0,
                     color=STATE_COLORS[st], alpha=0.8, rasterized=True)
    embedding_axes(ax_a)
    ax_a.set_aspect("auto")
    ax_a.set_xlabel("Component 1", labelpad=1.5, fontsize=5.0)
    ax_a.set_ylabel("Component 2", labelpad=1.5, fontsize=5.0)
    letters.append(ax_a)

    axes_b = [fig.add_subplot(row1[i + 1]) for i in range(len(GROUP_ORDER))]
    for ax, g in zip(axes_b, GROUP_ORDER):
        sub = traj[traj["Newgroup6"] == g]
        ax.scatter(traj["Component1"], traj["Component2"], s=0.5, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        for st in STATES:
            part = sub[sub["State"] == st]
            ax.scatter(part["Component1"], part["Component2"], s=0.7, linewidths=0,
                       color=STATE_COLORS[st], alpha=0.85, rasterized=True)
        embedding_axes(ax)
        ax.set_aspect("auto")
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.2, pad=1.2)
    letters.append(axes_b[0])

    state_handles = [Line2D([], [], marker="o", linestyle="none", markersize=2.4,
                            markerfacecolor=STATE_COLORS[s], markeredgecolor="none",
                            label=f"State {s}") for s in STATES]
    ax_a.legend(handles=state_handles, loc="upper left", bbox_to_anchor=(-0.02, 1.04),
                fontsize=5.0, handletextpad=0.3, labelspacing=0.35, borderpad=0.15)

    # ---- row 2: c (state occupancy per sample) ----------------------------
    freq = pd.read_csv(T04 / "State_frequency_by_sample.csv", dtype={"State": str})
    full = (freq.set_index(["orig.ident", "State"])["prop"]
            .unstack(fill_value=0.0).stack().rename("prop").reset_index())
    meta = freq[["orig.ident", "Newgroup6"]].drop_duplicates()
    full = full.merge(meta, on="orig.ident")
    full = full[full["Newgroup6"].isin(GROUP_ORDER)]

    grid_c = outer[1].subgridspec(1, len(STATES), wspace=0.50)
    axes_c = [fig.add_subplot(grid_c[i]) for i in range(len(STATES))]
    for idx, (ax, st) in enumerate(zip(axes_c, STATES)):
        sub = full[full["State"] == st]
        wide = sub.pivot_table(index="orig.ident", columns="Newgroup6", values="prop")
        cols = {g: wide[g].dropna().to_numpy() for g in GROUP_ORDER if g in wide}
        P.group_boxplots([ax], sub.assign(value=sub["prop"]), ["value"], "Newgroup6",
                         GROUP_ORDER, {"value": ""}, ylabel="Occupancy",
                         title_fontsize=5.2)
        arrays = [sub.loc[sub["Newgroup6"] == g, "prop"].to_numpy() for g in GROUP_ORDER]
        arrays = [a for a in arrays if len(a)]
        p = stats.kruskal(*arrays).pvalue if len(arrays) >= 2 else np.nan
        ax.set_title(f"State {st}  {significance_stars(p) or 'ns'}", fontsize=5.2,
                     pad=2.0)
        if idx:
            ax.set_ylabel("")
    letters.append(axes_c[0])
    NOTES.append("panel c: fraction of each sample's CD8+ cells assigned to a state; "
                 "Kruskal-Wallis across the four groups")

    # ---- row 3: d | e -----------------------------------------------------
    # Leading column is a gutter for the subtype names.
    row3 = outer[2].subgridspec(1, 6, width_ratios=[0.30, 0.62, 0.030, 0.14, 1.0, 1.0],
                               wspace=0.26)
    ctxs = pd.read_csv(T04 / "celltype_x_State_frequency.csv", dtype={"State": str})
    mat_d = (ctxs.pivot_table(index="celltype_new", columns="State", values="frac")
             .reindex(index=SUBTYPE_ORDER, columns=STATES).fillna(0.0))
    ax_d = fig.add_subplot(row3[1])
    im_d = ax_d.imshow(mat_d.to_numpy(), cmap=EXPR_CMAP, aspect="auto", vmin=0)
    ax_d.set_xticks(range(len(STATES)))
    ax_d.set_xticklabels([f"S{s}" for s in STATES])
    ax_d.set_yticks(range(len(SUBTYPE_ORDER)))
    ax_d.set_yticklabels([SUBTYPE_LABEL[s] for s in SUBTYPE_ORDER])
    ax_d.set_xlabel("State", labelpad=1.5)
    ax_d.tick_params(length=0, pad=1.0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    cax_d = fig.add_subplot(row3[2])
    cb = fig.colorbar(im_d, cax=cax_d)
    cb.ax.set_title("Fraction", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    letters.append(ax_d)

    binned = pd.read_csv(DATA / "fig4_pseudotime_binned_expr.csv")
    ax_e1 = fig.add_subplot(row3[4])
    P.pseudotime_lines(ax_e1, binned[binned["class"] == "Homeostatic"], "gene",
                       "bin", "mean_expr", HOME_COLOR)
    ax_e1.set_title("Homeostatic genes", fontsize=5.4, pad=2.0)
    ax_e2 = fig.add_subplot(row3[5])
    P.pseudotime_lines(ax_e2, binned[binned["class"] == "Progression"], "gene",
                       "bin", "mean_expr", PROG_COLOR)
    ax_e2.set_title("Progression genes", fontsize=5.4, pad=2.0)
    ax_e2.set_ylabel("")
    letters.append(ax_e1)
    NOTES.append("panel e: top 10 genes by |Spearman| against pseudotime in each "
                 "direction; thin lines are single genes, the thick line their mean, "
                 "over 20 equal-count pseudotime bins")

    # ---- row 4: f | g -----------------------------------------------------
    # Leading column is a gutter for the "subtype | state" row names.
    row4 = outer[3].subgridspec(1, 6,
                               width_ratios=[0.26, 1.10, 0.030, 0.15, 0.98, 0.030],
                               wspace=0.22)
    per_sample = (traj.groupby(["orig.ident", "Newgroup6", "celltype_new", "State"])
                  .size().rename("n").reset_index())
    totals = traj.groupby("orig.ident").size().rename("total")
    per_sample = per_sample.join(totals, on="orig.ident")
    per_sample["freq"] = per_sample["n"] / per_sample["total"]
    combo_mean = (per_sample.groupby(["celltype_new", "State", "Newgroup6"])["freq"]
                  .mean().unstack("Newgroup6").reindex(columns=GROUP_ORDER))
    combo_mean = combo_mean.reindex(
        [(ct, st) for ct in SUBTYPE_ORDER for st in STATES]).dropna(how="all")
    combo_mean = combo_mean.fillna(0.0)
    row_labels_f = [f"{SUBTYPE_SHORT[ct]} | S{st}" for ct, st in combo_mean.index]
    mat_f = combo_mean.to_numpy()
    half = int(np.ceil(len(mat_f) / 2))
    cax_f = fig.add_subplot(row4[2])
    ax_f = P.zscore_heatmap(fig, row4[1], cax_f,
                            _row_scale(mat_f), row_labels_f,
                            [GROUP_LABELS[g] for g in GROUP_ORDER],
                            [list(range(half)), list(range(half, len(mat_f)))],
                            gene_fontsize=5.0, wspace=1.30, italic=False)
    letters.append(ax_f)
    NOTES.append(f"panel f: {len(mat_f)} subtype-by-state combinations present in "
                 "the data; per-sample frequency averaged within group, then "
                 "z-scored across the four groups")

    mean_expr = pd.read_csv(XR / f"{STATE_PREFIX}_top_marker_mean_expr.csv",
                            index_col=0)
    mat_g = mean_expr.to_numpy()
    z_g = _row_scale(mat_g)
    genes_g = list(mean_expr.index)
    block = int(np.ceil(len(genes_g) / 4))
    blocks_g = [list(range(i, min(i + block, len(genes_g))))
                for i in range(0, len(genes_g), block)]
    cax_g = fig.add_subplot(row4[5])
    ax_g = P.zscore_heatmap(fig, row4[4], cax_g, z_g, genes_g,
                            [c.replace("State", "S") for c in mean_expr.columns],
                            blocks_g, gene_fontsize=5.0, wspace=1.05)
    letters.append(ax_g)
    NOTES.append(f"panel g: {len(genes_g)} state-defining markers, mean expression "
                 "per state z-scored across states")

    # ---- row 5: h | i -----------------------------------------------------
    # Leading column is a gutter for the KEGG set names.
    row5 = outer[4].subgridspec(1, 6,
                               width_ratios=[0.60, 0.30, 0.026, 0.17, 1.05, 0.028],
                               wspace=0.30)
    k2 = pd.read_csv(XR / f"{STATE_PREFIX}_State2_KEGG.csv")
    k4 = pd.read_csv(XR / f"{STATE_PREFIX}_State4_KEGG.csv")
    ax_h = fig.add_subplot(row5[1])
    cax_h = fig.add_subplot(row5[2])
    lax_h = fig.add_subplot(row5[3])
    kept = P.kegg_dotplot(fig, ax_h, cax_h, lax_h, [k2, k4], ["State 2", "State 4"],
                          n_top=9)
    letters.append(ax_h)
    NOTES.append(f"panel h: union of the 9 most significant KEGG sets per state "
                 f"({len(kept)} sets); dot area is the gene count, colour the "
                 "adjusted p value")

    beam = pd.read_csv(XR / f"{BEAM_PREFIX}_representative_heatmap_matrix.csv",
                       index_col=0)
    segments = ["Pre-branch" if c.startswith("Pre_") else
                "Fate1_State4" if c.startswith("Fate1_") else "Fate2_State3"
                for c in beam.columns]
    inner_i = row5[4].subgridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.06)
    sax_i = fig.add_subplot(inner_i[0])
    ax_i = fig.add_subplot(inner_i[1])
    cax_i = fig.add_subplot(row5[5])
    P.segment_heatmap(fig, ax_i, cax_i, beam.to_numpy(), list(beam.index), segments,
                      BRANCH_COLORS, sax=sax_i, segment_labels=BRANCH_LABEL)
    letters.append(sax_i)
    NOTES.append("panel i: branch-dependent genes at branch point 1, ordered by "
                 "pseudotime within the pre-branch segment and each fate")

    # ---- row 6: j ---------------------------------------------------------
    curves = pd.read_csv(XR / f"{BEAM_PREFIX}_branched_pseudotime_curves.csv")
    genes_j = list(dict.fromkeys(curves["gene"]))
    grid_j = outer[5].subgridspec(1, len(genes_j), wspace=0.42)
    axes_j = [fig.add_subplot(grid_j[i]) for i in range(len(genes_j))]
    for idx, (ax, gene) in enumerate(zip(axes_j, genes_j)):
        sub = curves[curves["gene"] == gene]
        for br, color in BRANCH_COLORS.items():
            part = sub[sub["branch"] == br].sort_values("pseudotime")
            ax.plot(part["pseudotime"], part["expr"], color=color, linewidth=0.7)
        ax.set_title(gene, fontsize=5.4, style="italic", pad=2.0)
        ax.set_xlabel("Pseudotime", labelpad=1.5)
        ax.tick_params(length=1.2, pad=1.0)
        if idx == 0:
            ax.set_ylabel("Expression", labelpad=1.5)
    letters.append(axes_j[0])
    NOTES.append("panel j: branch colours match the segment bar of panel i "
                 "(grey pre-branch, red State 4 fate, blue State 3 fate)")

    for ax, lab in zip(letters, "abcdefghij"):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure4")
    (OUT / "Figure4_provenance.txt").write_text(
        "Figure 4 | Pseudotemporal ordering of CD8+ cell states\n"
        "Rendered in Python (matplotlib). R used only to export DDRTree coordinates\n"
        "and pseudotime-binned expression from the monocle2 CDS.\n\n"
        "Note: panels a-f come from the HC-rooted four-group trajectory, while\n"
        "panels g-j reuse the earlier state-specific and BEAM analyses of the same\n"
        "CDS without the HC re-rooting, exactly as in the original figure.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


def _row_scale(mat):
    return np.nan_to_num((mat - mat.mean(axis=1, keepdims=True))
                         / (mat.std(axis=1, keepdims=True) + 1e-9))


if __name__ == "__main__":
    main()
