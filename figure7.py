"""Figure 7 | NK cells lose the homeostatic programme and gain an oxidative,
antigen-presenting one along the disease continuum.

Core conclusion: the NK compartment shifts away from the resting NCAM1/TCF7
subsets towards FCGR3A/GZMB effectors, and the pathways that separate the
homeostatic from the progressive axis reverse sign between them: oxidative
phosphorylation and phagosome genes rise while the signalling programmes that
mark homeostasis fall. The same ordering is recovered without supervision by a
monocle2 trajectory over the NK cells.

Panels
  a  NK embedding resolved into six subtypes
  b  Subtype-defining marker expression
  c  Per-patient subtype proportions across groups
  d  KEGG pathways whose enrichment reverses between the two axes
  e  Leading-edge genes of those pathways, by group
  f  Monocle2 trajectory coloured by pseudotime
  g  The same trajectory split by group and coloured by state
  h  State-defining markers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    embedding_axes,
    panel_label,
    save_figure,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig7"
FIG07 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig07")
OUT = ROOT / "Figure7"

SUBTYPE_ORDER = [
    "NK_NCAM1_TCF7_SELL_IL7R", "NK_NCAM1_DAB2", "NK_CD69_TNFSF14_IRF8",
    "NK_IFNG_IRF4", "NK_AKR1C3_FCGR3A_GZMB", "NK_MKI67",
]
SUBTYPE_LABEL = {s: s.replace("NK_", "").replace("_", " ") for s in SUBTYPE_ORDER}
SUBTYPE_LABEL_WRAP = dict(SUBTYPE_LABEL, **{
    "NK_NCAM1_TCF7_SELL_IL7R": "NCAM1 TCF7\nSELL IL7R",
    "NK_CD69_TNFSF14_IRF8": "CD69 TNFSF14\nIRF8",
    "NK_AKR1C3_FCGR3A_GZMB": "AKR1C3 FCGR3A\nGZMB",
})
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER, [
    "#66C2A5", "#8DA0CB", "#A6D854", "#FFD92F", "#E78AC3", "#E5C494",
]))

MARKERS = ["NCAM1", "TCF7", "SELL", "IL7R", "DAB2", "CD69", "TNFSF14", "IRF8",
           "IFNG", "IRF4", "AKR1C3", "FCGR3A", "GZMB", "PRF1", "GNLY", "NKG7",
           "MKI67"]

PATHWAY_SHORT = {
    "Oxidative phosphorylation": "Oxidative phosphorylation",
    "Parathyroid hormone synthesis, secretion and action": "Parathyroid hormone signalling",
    "Non-alcoholic fatty liver disease": "Non-alcoholic fatty liver",
}

STATE_COLORS = {
    "1": "#E15759", "2": "#B07AA1", "3": "#76B7B2", "4": "#59A14F",
    "5": "#4E79A7", "6": "#9C755F", "7": "#F28E2B",
}

NOTES: list[str] = []


def main() -> None:
    apply_publication_style()

    cells = pd.read_csv(DATA / "fig7_cells.csv")
    prop = pd.read_csv(DATA / "fig7_patient_proportions.csv")
    traj = pd.read_csv(DATA / "fig7_trajectory.csv")
    edges = pd.read_csv(DATA / "fig7_tree_edges.csv")
    nes = pd.read_csv(FIG07 / "panel_h_HP_reverse_pathways_top.csv")
    lead = pd.read_csv(FIG07 / "panel_i_HP_leading_genes.csv")
    marker_expr = pd.read_csv(DATA / "fig7_state_marker_expr.csv")

    traj["State"] = traj["State"].astype(str)
    NOTES.append(f"panels a-c,e: {len(cells):,} NK cells over {GROUP_ORDER} "
                 "(IM_M and MH_CD56 excluded to keep the four-group continuum); "
                 "embedding is the harmony-integrated UMAP")

    # A handful of cells sit far outside the manifold and would otherwise
    # compress the whole embedding into a corner of the panel.
    keep = np.ones(len(cells), dtype=bool)
    for col in ("UMAP1", "UMAP2"):
        lo, hi = np.percentile(cells[col], [0.2, 99.8])
        keep &= cells[col].between(lo, hi).to_numpy()
    embed = cells[keep]
    NOTES.append(f"panel a: {(~keep).sum()} of {len(cells):,} cells fall outside "
                 "the 0.2-99.8 percentile of either embedding axis and are not "
                 "drawn; they are included in every other panel")
    NOTES.append(f"panels f-h: monocle2 ordering of {len(traj):,} NK cells "
                 f"(downsampled CDS) resolved into {traj['State'].nunique()} states")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.30))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.58, 1.02, 1.60, 1.34, 1.30],
                             left=0.070, right=0.960, top=0.972, bottom=0.048,
                             hspace=0.58)

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.30, 0.016, 0.17],
                                wspace=0.10)
    ax_a = fig.add_subplot(row1[0])
    P.umap_by_category(ax_a, embed, "celltype", SUBTYPE_COLORS,
                       SUBTYPE_LABEL_WRAP, size=0.6, min_dist_frac=0.62)
    ax_a.set_title(f"{len(cells):,} NK cells  |  6 subtypes", fontsize=5.2,
                   color=NEUTRAL_MID, pad=1.5)

    ax_b = fig.add_subplot(row1[1])
    cax_b = fig.add_subplot(row1[2])
    side_b = row1[3].subgridspec(2, 1, hspace=0.25)
    lax_b = fig.add_subplot(side_b[1])
    genes_b = [g for g in MARKERS if g in cells.columns]
    P.dotplot(fig, ax_b, cax_b, lax_b, cells, genes_b, "celltype", SUBTYPE_ORDER,
              SUBTYPE_LABEL, genes_on_x=True, size_scale=0.32)

    # ---- row 2: c ---------------------------------------------------------
    grid_c = outer[1].subgridspec(1, 6, wspace=0.55)
    axes_c = [fig.add_subplot(grid_c[i]) for i in range(6)]
    NOTES.extend(P.proportion_boxplots(axes_c, prop, SUBTYPE_ORDER, SUBTYPE_LABEL,
                                       ncol=6))

    # ---- row 3: d | e -----------------------------------------------------
    # The leading column is an empty gutter: it gives the pathway names room to
    # sit outside the plot area.
    row3 = outer[2].subgridspec(1, 6,
                                width_ratios=[0.50, 0.36, 0.026, 0.10, 1.62, 0.026],
                                wspace=0.24)
    left_d = row3[1].subgridspec(2, 1, height_ratios=[1.0, 0.34], hspace=0.62)
    ax_d = fig.add_subplot(left_d[0])
    lax_e = fig.add_subplot(left_d[1])
    strip_axes(lax_e)
    cax_d = fig.add_subplot(row3[2])
    nes_mat = nes[["NES_homeostatic", "NES_progressive"]].to_numpy()
    order_d = np.argsort(nes["NES_diff"].to_numpy())
    nes_mat = nes_mat[order_d]
    terms = [PATHWAY_SHORT.get(t, t) for t in nes["term"].to_numpy()[order_d]]
    vmax_d = float(np.abs(nes_mat).max())
    im_d = ax_d.imshow(nes_mat, cmap=DIVERGING_CMAP, aspect="auto",
                       vmin=-vmax_d, vmax=vmax_d)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Homeostatic", "Progressive"], rotation=40, ha="right",
                         rotation_mode="anchor")
    ax_d.set_yticks(range(len(terms)))
    ax_d.set_yticklabels([P._shorten(t, 26) for t in terms], fontsize=5.0)
    ax_d.tick_params(length=0, pad=1.0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    cb_d = fig.colorbar(im_d, cax=cax_d)
    cb_d.ax.set_title("NES", fontsize=5.0, pad=2.0)
    cb_d.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb_d.outline.set_linewidth(0.4)

    # Keep only pathways carrying enough leading-edge genes to read as a block,
    # and cap each at its six strongest genes so the labels stay legible.
    lead = lead[lead["gene"].isin(cells.columns)].copy()
    sizes = lead.groupby("pathway")["gene"].size()
    pathway_order = sizes[sizes >= 3].sort_values(ascending=False).index.tolist()
    lead = lead[lead["pathway"].isin(pathway_order)].copy()
    lead["pathway"] = pd.Categorical(lead["pathway"], pathway_order)
    lead = (lead.sort_values(["pathway", "score"], ascending=[True, False])
            .groupby("pathway", observed=True).head(6))
    genes_e = lead["gene"].drop_duplicates().tolist()
    mat_e = np.array([[cells.loc[cells["group"] == g, gene].mean()
                       for g in GROUP_ORDER] for gene in genes_e])
    z_e = np.nan_to_num((mat_e - mat_e.mean(axis=1, keepdims=True))
                        / (mat_e.std(axis=1, keepdims=True) + 1e-9))
    path_colors = dict(zip(pathway_order, [
        "#4C7FB8", "#C0392B", "#7BAA5B", "#E28E2C", "#8E7CC3", "#59A14F",
        "#B07AA1", "#76B7B2", "#E5C494", "#9C755F", "#BAB0AC",
    ][:len(pathway_order)]))
    row_groups_e = lead.drop_duplicates("gene")["pathway"].astype(str).tolist()
    cax_e = fig.add_subplot(row3[5])
    anchor_e = P.annotated_block_heatmap(
        fig, row3[4], cax_e, z_e, genes_e,
        [GROUP_LABELS[g] for g in GROUP_ORDER], row_groups_e,
        {str(k): v for k, v in path_colors.items()}, nblocks=3, wspace=1.30)
    handles_e = [Line2D([], [], marker="s", linestyle="none", markersize=2.6,
                        markerfacecolor=path_colors[p], markeredgecolor="none",
                        label=P._shorten(PATHWAY_SHORT.get(p, p), 24))
                 for p in pathway_order]
    leg_e = lax_e.legend(handles=handles_e, loc="upper left", ncol=1,
                         fontsize=5.0, handletextpad=0.35, labelspacing=0.30,
                         borderpad=0.1, borderaxespad=0.0, title="Pathway")
    leg_e.get_title().set_fontsize(5.0)
    NOTES.append(f"panel d: {len(terms)} KEGG pathways whose GSEA enrichment "
                 "reverses sign between the homeostatic and progressive axes")
    NOTES.append(f"panel e: {len(genes_e)} leading-edge genes of those pathways; "
                 "group-level mean expression z-scored across the four groups")

    # ---- row 4: f | g -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 6,
                                width_ratios=[1.0, 0.020, 0.30, 3.0, 0.02, 0.20],
                                wspace=0.10)
    ax_f = fig.add_subplot(row4[0])
    cax_f = fig.add_subplot(row4[1])
    for _, e in edges.iterrows():
        ax_f.plot([e["x"], e["xend"]], [e["y"], e["yend"]], color="#1A1A1A",
                  linewidth=0.25, zorder=2)
    sc_f = ax_f.scatter(traj["Component1"], traj["Component2"], c=traj["Pseudotime"],
                        cmap="plasma", s=0.7, linewidths=0, rasterized=True, zorder=1)
    strip_axes(ax_f)
    ax_f.set_xlabel("Component 1", fontsize=5.0, labelpad=1.0)
    ax_f.set_ylabel("Component 2", fontsize=5.0, labelpad=1.0)
    cb_f = fig.colorbar(sc_f, cax=cax_f)
    cb_f.ax.set_title("Pseudo-\ntime", fontsize=5.0, pad=2.0)
    cb_f.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb_f.outline.set_linewidth(0.4)

    grid_g = row4[3].subgridspec(1, len(GROUP_ORDER), wspace=0.06)
    axes_g = []
    for i, g in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(grid_g[i])
        axes_g.append(ax)
        ax.scatter(traj["Component1"], traj["Component2"], s=0.5, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        sub = traj[traj["group"] == g]
        ax.scatter(sub["Component1"], sub["Component2"], s=0.6, linewidths=0,
                   color=[STATE_COLORS[s] for s in sub["State"]], alpha=0.85,
                   rasterized=True)
        strip_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.2, pad=1.2)
    handles_g = [Line2D([], [], marker="o", linestyle="none", markersize=2.2,
                        markerfacecolor=STATE_COLORS[s], markeredgecolor="none",
                        label=s) for s in sorted(STATE_COLORS)]
    lax_g = fig.add_subplot(row4[5])
    strip_axes(lax_g)
    leg_g = lax_g.legend(handles=handles_g, title="State", loc="center left",
                         fontsize=5.0, handletextpad=0.35, labelspacing=0.30,
                         borderpad=0.15)
    leg_g.get_title().set_fontsize(5.0)
    NOTES.append("panel g: grey points show all trajectory cells as context; "
                 "coloured points are that group only")

    # ---- row 5: h ---------------------------------------------------------
    row5 = outer[4].subgridspec(1, 2, width_ratios=[1.0, 0.018], wspace=0.06)
    states = sorted(marker_expr["state"].astype(str).unique(), key=int)
    marker_expr["state"] = marker_expr["state"].astype(str)
    marker_expr["marker_state"] = marker_expr["marker_state"].astype(str)
    genes_h = (marker_expr.drop_duplicates("gene")
               .sort_values(["marker_state", "gene"], key=lambda s: s.map(
                   lambda v: int(v) if str(v).isdigit() else v)))
    gene_list = genes_h["gene"].tolist()
    pivot = marker_expr.pivot_table(index="gene", columns="state",
                                    values="mean_expr").reindex(gene_list)[states]
    mat_h = pivot.to_numpy()
    z_h = np.nan_to_num((mat_h - mat_h.mean(axis=1, keepdims=True))
                        / (mat_h.std(axis=1, keepdims=True) + 1e-9))
    cax_h = fig.add_subplot(row5[1])
    anchor_h = P.annotated_block_heatmap(
        fig, row5[0], cax_h, z_h, gene_list, states,
        genes_h["marker_state"].tolist(), STATE_COLORS, nblocks=3, wspace=1.10)
    anchor_h.set_title("State-defining markers    (columns: monocle2 state)",
                       fontsize=5.2, pad=3.0, loc="left")
    NOTES.append(f"panel h: {len(gene_list)} state-defining markers, colour strip "
                 "gives the state each marker defines; columns are the states in "
                 "order")

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (axes_c[0], "c"), (ax_d, "d"),
                    (anchor_e, "e"), (ax_f, "f"), (axes_g[0], "g"),
                    (anchor_h, "h")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure7")
    (OUT / "Figure7_provenance.txt").write_text(
        "Figure 7 | NK cells lose the homeostatic programme and gain an "
        "oxidative, antigen-presenting one\nRendered in Python (matplotlib); R "
        "used only to export data from the Seurat and monocle objects.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
