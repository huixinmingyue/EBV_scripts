"""Figure 10 | B cells lose their naive IgM/CXCR4 pool and move towards
activated and plasmablast states along the disease continuum.

Core conclusion: the B-cell compartment is reorganised in the same direction as
the T and NK compartments. The naive IgM/CXCR4 pool shrinks while IL4R/CD24/CD22
activated and CD38/MZB1 plasmablast subsets expand; the pathways that rise from
IM through MH-CD4 to HLH are the oxidative and inflammatory ones, and the KEGG
programmes separating the homeostatic from the progressive axis reverse sign
between them. An unsupervised monocle2 ordering recovers the same progression.

Panels
  a  B-cell embedding resolved into eight subtypes
  b  Subtype-defining marker expression
  c  Per-patient subtype proportions across groups
  d  The same embedding split by group
  e  Signalling pathways rising across IM to MH-CD4 to HLH
  f  Signalling pathways falling across the same continuum
  g  KEGG pathways whose enrichment reverses between the two axes
  h  Monocle2 trajectory coloured by pseudotime
  i  The same trajectory split by group and coloured by state
  j  Per-patient state occupancy across groups
  k  Genes with a monotonic trend across the continuum

The original figure carried the marker violin plots and the marker dot plot as
two panels over the same genes and the same subtypes; they are consolidated
into b.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import panels as P
from nature_style import (
    DIVERGING_CMAP,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig10"
FIG10 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig10")
OUT = ROOT / "Figure10"

SUBTYPE_ORDER = ["BC_IgM_CXCR4", "BC_IL4R_CD24_CD22", "BC_IL4R_CD22",
                 "BC_IL4R_CD22_CCR7", "BC_AIM2_CD70", "BC_CD27_CD70_BCL3",
                 "BC_SOX5_TNFRSF1B", "BC_CD38_MZB1"]
SUBTYPE_LABEL = {s: s.replace("BC_", "").replace("_", " ") for s in SUBTYPE_ORDER}
SUBTYPE_LABEL_WRAP = dict(SUBTYPE_LABEL, **{
    "BC_IL4R_CD24_CD22": "IL4R CD24\nCD22",
    "BC_IL4R_CD22_CCR7": "IL4R CD22\nCCR7",
    "BC_CD27_CD70_BCL3": "CD27 CD70\nBCL3",
    "BC_SOX5_TNFRSF1B": "SOX5\nTNFRSF1B",
})
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER, [
    "#4E79A7", "#76B7B2", "#8DA0CB", "#A6D854", "#59A14F", "#E28E2C",
    "#B07AA1", "#C0392B",
]))

MARKERS = ["IGHM", "IGHD", "CXCR4", "IL4R", "CD24", "CD22", "CCR7", "AIM2",
           "CD70", "CD27", "BCL3", "SOX5", "TNFRSF1B", "CD38", "MZB1", "XBP1",
           "PRDM1"]

TREND_GROUPS = ["IM", "MH_CD4", "HLH"]
STATE_COLORS = {
    "1": "#4E79A7", "2": "#76B7B2", "3": "#59A14F", "4": "#E28E2C",
    "5": "#C0392B", "6": "#B07AA1", "7": "#9C755F",
}
PATHWAY_SHORT = {
    "Antigen processing and presentation": "Antigen processing",
    "Transcriptional misregulation in cancer": "Transcriptional misreg. in cancer",
    "NOD-like receptor signaling pathway": "NOD-like receptor signalling",
    "NF-kappa B signaling pathway": "NF-kappa B signalling",
    "T cell receptor signaling pathway": "T-cell receptor signalling",
    "TNF signaling pathway": "TNF signalling",
}

NOTES: list[str] = []


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    return np.nan_to_num((mat - mat.mean(axis=1, keepdims=True))
                         / (mat.std(axis=1, keepdims=True) + 1e-9))


def main() -> None:
    apply_publication_style()

    cells = pd.read_csv(DATA / "fig10_cells.csv")
    prop = pd.read_csv(DATA / "fig10_patient_proportions.csv").rename(
        columns={"orig.ident": "sample"})
    traj = pd.read_csv(DATA / "fig10_trajectory.csv")
    edges = pd.read_csv(DATA / "fig10_tree_edges.csv")
    trend = pd.read_csv(FIG10 / "TC_EBV_05_B_monotonic" / "B_pathway_trend.csv")
    rev = pd.read_csv(FIG10 / "TC_EBV_03_B_reverse_gsea" / "B_KEGG_reverse_pathways.csv")
    inc = pd.read_csv(FIG10 / "TC_EBV_05_B_monotonic" / "B_monotonic_increasing.csv")
    dec = pd.read_csv(FIG10 / "TC_EBV_05_B_monotonic" / "B_monotonic_decreasing.csv")

    traj["State"] = traj["State"].astype(str)
    NOTES.append(f"panels a-d: {len(cells):,} B cells over {GROUP_ORDER} "
                 "(IM_M and MH_CD56 excluded to keep the four-group continuum)")

    # A handful of cells sit far outside the manifold and would otherwise squeeze
    # every subtype into the middle of the panel.
    keep = np.ones(len(cells), dtype=bool)
    for col in ("UMAP1", "UMAP2"):
        lo, hi = np.percentile(cells[col], [0.2, 99.8])
        keep &= cells[col].between(lo, hi).to_numpy()
    embed = cells[keep]
    NOTES.append(f"panels a,d: {(~keep).sum()} of {len(cells):,} cells fall outside "
                 "the 0.2-99.8 percentile of either embedding axis and are not "
                 "drawn; they are included in every other panel")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.10))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.48, 1.00, 1.50, 1.62, 1.30],
                             left=0.068, right=0.962, top=0.972, bottom=0.046,
                             hspace=0.52)

    # ---- row 1: a | b -----------------------------------------------------
    # The third column is an empty gutter between the subtype key and the dot
    # plot's own row labels.
    row1 = outer[0].subgridspec(1, 6,
                                width_ratios=[0.82, 0.28, 0.40, 1.18, 0.016, 0.17],
                                wspace=0.10)
    ax_a = fig.add_subplot(row1[0])
    # Eight subtypes sit in one small, tight manifold here; direct labels would
    # cover the clusters they name, so this panel uses a key.
    P.umap_by_category(ax_a, embed, "celltype", SUBTYPE_COLORS, size=0.5,
                       label=False)
    ax_a.set_title(f"{len(cells):,} B cells  |  8 subtypes", fontsize=5.2,
                   color=NEUTRAL_MID, pad=1.5)
    lax_a = fig.add_subplot(row1[1])
    strip_axes(lax_a)
    lax_a.legend(handles=[Line2D([], [], marker="o", linestyle="none",
                                 markersize=2.2,
                                 markerfacecolor=SUBTYPE_COLORS[s],
                                 markeredgecolor="none", label=SUBTYPE_LABEL[s])
                          for s in SUBTYPE_ORDER],
                 loc="center left", fontsize=5.0, handletextpad=0.35,
                 labelspacing=0.42, borderpad=0.1, borderaxespad=0.0)

    ax_b = fig.add_subplot(row1[3])
    cax_b = fig.add_subplot(row1[4])
    side_b = row1[5].subgridspec(2, 1, hspace=0.25)
    lax_b = fig.add_subplot(side_b[1])
    genes_b = [g for g in MARKERS if g in cells.columns]
    P.dotplot(fig, ax_b, cax_b, lax_b, cells, genes_b, "celltype", SUBTYPE_ORDER,
              SUBTYPE_LABEL, genes_on_x=True, size_scale=0.30)

    # ---- row 2: c ---------------------------------------------------------
    grid_c = outer[1].subgridspec(1, 8, wspace=0.62)
    axes_c = [fig.add_subplot(grid_c[i]) for i in range(8)]
    NOTES.extend(P.proportion_boxplots(axes_c, prop, SUBTYPE_ORDER, SUBTYPE_LABEL,
                                       ncol=8))

    # ---- row 3: d | e | f -------------------------------------------------
    row3 = outer[2].subgridspec(1, 4, width_ratios=[0.80, 0.16, 1.05, 1.05],
                                wspace=0.14)
    grid_d = row3[0].subgridspec(2, 2, wspace=0.06, hspace=0.30)
    axes_d = []
    for i, g in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(grid_d[i // 2, i % 2])
        axes_d.append(ax)
        ax.scatter(embed["UMAP1"], embed["UMAP2"], s=0.30, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        sub = embed[embed["group"] == g]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.35, linewidths=0, alpha=0.85,
                   color=[SUBTYPE_COLORS[c] for c in sub["celltype"]],
                   rasterized=True)
        strip_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.0, pad=1.0)
    NOTES.append("panel d: grey points show all B cells as context; coloured "
                 "points are that group only, keeping the subtype palette of a")

    trend_long = trend.melt(id_vars=["pathway", "class"],
                            value_vars=["mean_IM", "mean_MH_CD4", "mean_HLH"],
                            var_name="group", value_name="mean_expr")
    trend_long["group"] = trend_long["group"].str.replace("mean_", "", regex=False)
    for idx, (cls, title, ax_key) in enumerate(
            [("monotonic_increasing", "Rising across IM to MH-CD4 to HLH", "e"),
             ("monotonic_decreasing", "Falling across the same continuum", "f")]):
        ax = fig.add_subplot(row3[2 + idx])
        sub = trend[trend["class"] == cls].copy()
        span = (sub["mean_HLH"] - sub["mean_IM"]).abs() / (sub["mean_IM"] + 1e-6)
        sub = sub.assign(_span=span).sort_values("_span", ascending=False).head(6)
        order = sub.sort_values("mean_HLH", ascending=False)["pathway"].tolist()
        P.grouped_hbars(ax, trend_long, "pathway", "group", "mean_expr", order,
                        TREND_GROUPS,
                        {g: GROUP_COLORS[g] for g in TREND_GROUPS},
                        xlabel="Mean expression", max_label_chars=28)
        ax.set_title(title, fontsize=5.2, pad=2.0, loc="left")
        if ax_key == "e":
            ax_e = ax
        else:
            ax_f = ax
            # The two shortest pathways leave the lower right of this panel empty,
            # which is the only place in the row that fits the key.
            ax.legend(handles=[Patch(facecolor=GROUP_COLORS[g], edgecolor="none",
                                     label=GROUP_LABELS[g]) for g in TREND_GROUPS],
                      loc="lower right", fontsize=5.0, handlelength=1.0,
                      handletextpad=0.35, labelspacing=0.25, borderpad=0.25,
                      frameon=False)
    n_inc = (trend["class"] == "monotonic_increasing").sum()
    n_dec = (trend["class"] == "monotonic_decreasing").sum()
    NOTES.append("panels e,f: KEGG signalling pathways whose group-mean expression "
                 f"is strictly monotonic across IM, MH-CD4 and HLH ({n_inc} rising "
                 f"and {n_dec} falling of {len(trend)} pathways tested); up to six "
                 "per direction are shown, ranked by relative change from IM to HLH")
    NOTES.append("panels e,f use the pathway-level table (all genes in a pathway, "
                 "pathway mean required to be monotonic). The archived alternative "
                 "splits each pathway's genes into rising and falling subsets and "
                 "averages them separately, which lists every one of its pathways "
                 "in both directions and carries FDR = 1 throughout; it is drawn "
                 "in Figure 10 supplement 1 rather than here, and the two readings "
                 "are not comparable")

    # ---- row 4: g | h | i -------------------------------------------------
    # The leading column is an empty gutter, giving the pathway names in g room to
    # sit outside the plot area.
    row4 = outer[3].subgridspec(1, 7,
                                width_ratios=[0.32, 0.60, 0.022, 0.34, 0.72,
                                              0.020, 1.62],
                                wspace=0.16)
    # Keep the pathways with the widest split between the two axes: the panel is
    # about which programmes reverse, not about listing every reversal.
    rv = rev.reindex(rev["NES_diff"].abs().sort_values(ascending=False).index).head(14)
    rv = rv.sort_values("NES_diff")
    nes_mat = rv[["NES_homeostatic", "NES_progressive"]].to_numpy()
    ax_g = fig.add_subplot(row4[1])
    cax_g = fig.add_subplot(row4[2])
    vmax_g = float(np.abs(nes_mat).max())
    im_g = ax_g.imshow(nes_mat, cmap=DIVERGING_CMAP, aspect="auto", vmin=-vmax_g,
                       vmax=vmax_g)
    ax_g.set_xticks([0, 1])
    ax_g.set_xticklabels(["Homeostatic", "Progressive"], rotation=40, ha="right",
                         rotation_mode="anchor")
    ax_g.set_yticks(range(len(rv)))
    ax_g.set_yticklabels([P._shorten(PATHWAY_SHORT.get(t, t), 28)
                          for t in rv["term"]], fontsize=5.0)
    ax_g.tick_params(length=0, pad=1.0)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    cb_g = fig.colorbar(im_g, cax=cax_g)
    cb_g.ax.set_title("NES", fontsize=5.0, pad=2.0)
    cb_g.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb_g.outline.set_linewidth(0.4)
    NOTES.append(f"panel g: {len(rv)} of {len(rev)} KEGG pathways whose GSEA "
                 "enrichment reverses sign between the homeostatic (IM vs HC) and "
                 "progressive (HLH vs IM) axes, ranked by the size of the split")

    ax_h = fig.add_subplot(row4[4])
    cax_h = fig.add_subplot(row4[5])
    for _, e in edges.iterrows():
        ax_h.plot([e["x"], e["xend"]], [e["y"], e["yend"]], color="#1A1A1A",
                  linewidth=0.25, zorder=2)
    sc_h = ax_h.scatter(traj["Component1"], traj["Component2"],
                        c=traj["Pseudotime"], cmap="plasma", s=0.7, linewidths=0,
                        rasterized=True, zorder=1)
    strip_axes(ax_h)
    ax_h.set_xlabel("Component 1", fontsize=5.0, labelpad=1.0)
    ax_h.set_ylabel("Component 2", fontsize=5.0, labelpad=1.0)
    cb_h = fig.colorbar(sc_h, cax=cax_h)
    cb_h.ax.set_title("Pseudo-\ntime", fontsize=5.0, pad=2.0)
    cb_h.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb_h.outline.set_linewidth(0.4)

    grid_i = row4[6].subgridspec(1, len(GROUP_ORDER) + 2,
                                 width_ratios=[0.32] + [1.0] * len(GROUP_ORDER)
                                 + [0.30], wspace=0.06)
    axes_i = []
    for i, g in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(grid_i[i + 1])
        axes_i.append(ax)
        ax.scatter(traj["Component1"], traj["Component2"], s=0.5, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        sub = traj[traj["group"] == g]
        ax.scatter(sub["Component1"], sub["Component2"], s=0.6, linewidths=0,
                   color=[STATE_COLORS[s] for s in sub["State"]], alpha=0.85,
                   rasterized=True)
        strip_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}\nn = {len(sub):,}", fontsize=5.2, pad=1.2)
    lax_i = fig.add_subplot(grid_i[len(GROUP_ORDER) + 1])
    strip_axes(lax_i)
    leg_i = lax_i.legend(
        handles=[Line2D([], [], marker="o", linestyle="none", markersize=2.2,
                        markerfacecolor=STATE_COLORS[s], markeredgecolor="none",
                        label=s) for s in sorted(STATE_COLORS)],
        title="State", loc="center left", fontsize=5.0, handletextpad=0.35,
        labelspacing=0.30, borderpad=0.15, borderaxespad=0.0)
    leg_i.get_title().set_fontsize(5.0)
    NOTES.append(f"panels h-j: monocle2 ordering of {len(traj):,} B cells resolved "
                 f"into {traj['State'].nunique()} states; grey points in i show all "
                 "trajectory cells as context")

    # ---- row 5: j | k -----------------------------------------------------
    row5 = outer[4].subgridspec(1, 5, width_ratios=[0.86, 0.28, 0.18, 1.30, 0.022],
                                wspace=0.12)
    states = sorted(traj["State"].unique(), key=int)
    occ = (traj.groupby(["sample", "group", "State"]).size()
           .rename("n").reset_index())
    occ["frequency"] = occ["n"] / occ.groupby("sample")["n"].transform("sum") * 100
    wide = occ.pivot_table(index=["sample", "group"], columns="State",
                           values="frequency", fill_value=0.0).reset_index()
    wide = wide.rename(columns={"group": "Newgroup6"})
    pvals = {}
    for s in states:
        arms = [wide.loc[wide["Newgroup6"] == g, s].dropna().to_numpy()
                for g in GROUP_ORDER]
        arms = [a for a in arms if len(a) > 0]
        pvals[s] = stats.kruskal(*arms).pvalue if len(arms) >= 2 else np.nan
    ax_j = fig.add_subplot(row5[0])
    P.grouped_boxplots(ax_j, wide, states, "Newgroup6", GROUP_ORDER,
                       labels={s: s for s in states}, pvalues=pvals,
                       ylabel="State occupancy (%)", rotate=0)
    ax_j.set_xlabel("Monocle2 state", labelpad=1.5)
    ax_j.set_title("Per-patient state occupancy", fontsize=5.2, pad=2.0, loc="left")
    lax_j = fig.add_subplot(row5[1])
    strip_axes(lax_j)
    lax_j.legend(handles=[Patch(facecolor=GROUP_COLORS[g], edgecolor="none",
                                label=GROUP_LABELS[g]) for g in GROUP_ORDER],
                 loc="center left", fontsize=5.0, handlelength=1.0,
                 handletextpad=0.35, labelspacing=0.30, borderpad=0.1,
                 borderaxespad=0.0)
    NOTES.append(f"panel j: state occupancy per patient over "
                 f"{wide['sample'].nunique()} samples; Kruskal-Wallis across the "
                 "four groups")
    NOTES.append("panels h-j all come from cds_bcells_new_celltype_monocle2_"
                 "no_regress.rds. An older per-sample state table exists from a "
                 "separate ds2000 monocle run (10,471 cells, all six groups); "
                 "state numbers are not comparable between two independent DDRTree "
                 "fits, so it is not mixed with the embeddings drawn here")

    top_inc = inc.head(12).assign(regulation="Up")
    top_dec = dec.head(12).assign(regulation="Down")
    mono = pd.concat([top_inc, top_dec], ignore_index=True)
    mat_k = mono[["mean_IM", "mean_MH_CD4", "mean_HLH"]].to_numpy()
    z_k = zscore_rows(mat_k)
    cax_k = fig.add_subplot(row5[4])
    anchor_k = P.annotated_block_heatmap(
        fig, row5[3], cax_k, z_k, mono["gene"].tolist(),
        [GROUP_LABELS[g] for g in TREND_GROUPS], mono["regulation"].tolist(),
        {"Up": "#C0392B", "Down": "#4E79A7"}, nblocks=2, wspace=1.30,
        col_rotation=90)
    anchor_k.set_title("Monotonic genes across the continuum "
                       "(red strip: rising, blue: falling)", fontsize=5.2,
                       pad=3.0, loc="left")
    NOTES.append("panel k: the 12 strongest rising and 12 strongest falling genes "
                 f"of {len(inc)} monotonically increasing and {len(dec)} "
                 "monotonically decreasing genes, ranked by the per-cell Spearman "
                 "correlation with the ordered groups; each gene z-scored across "
                 "the three group means")

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (axes_c[0], "c"), (axes_d[0], "d"),
                    (ax_e, "e"), (ax_f, "f"), (ax_g, "g"), (ax_h, "h"),
                    (axes_i[0], "i"), (ax_j, "j"), (anchor_k, "k")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure10")
    (OUT / "Figure10_provenance.txt").write_text(
        "Figure 10 | B cells lose their naive IgM/CXCR4 pool and move towards "
        "activated and plasmablast states\nRendered in Python (matplotlib); R "
        "used only to export data from the Seurat and monocle objects.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
