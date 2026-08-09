"""Figure 8 | An NLRP3/HK2 inflammatory monocyte state expands with disease
severity and rewires monocyte-to-CD8 signalling.

Core conclusion: the monocyte compartment resolves into four states, and the
glycolytic inflammasome state MC_NLRP3_HK2 is the one that tracks disease. It
dominates HLH, carries the inflammatory and tryptophan-catabolising programme,
scales with serum ferritin, and is the source of a signalling shift towards the
exhausted GZMK/TIGIT/HAVCR2 CD8 subset.

Panels
  a  Monocyte embedding resolved into four states
  b  State-defining and effector marker expression
  c  The same embedding split by group
  d  Per-patient state proportions across groups
  e  Genes separating MC_NLRP3_HK2 from the other states
  f  The same genes across groups within MC_NLRP3_HK2
  g  Inflammasome and tryptophan-catabolism effectors by state
  h  MC_NLRP3_HK2 -> CD8 signalling strength by CD8 subtype
  i  Ligand-receptor pairs driving that shift
  j  Total interaction probability across the group continuum
  k  State proportion against serum ferritin

The original figure carried the marker dot plot and a marker violin plot as two
panels over the same genes and the same states; they are consolidated into b.
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
DATA = ROOT / "_data" / "fig8"
OUT = ROOT / "Figure8"

SUBTYPE_ORDER = ["MC_CD14_S100A8_CD163_RETN", "MC_NLRP3_HK2", "MC_CD16_CX3CR1",
                 "MC_SPIB_HLA-DRA_FLT3"]
SUBTYPE_LABEL = {
    "MC_CD14_S100A8_CD163_RETN": "CD14 S100A8 CD163 RETN",
    "MC_NLRP3_HK2": "NLRP3 HK2",
    "MC_CD16_CX3CR1": "CD16 CX3CR1",
    "MC_SPIB_HLA-DRA_FLT3": "SPIB HLA-DRA FLT3",
}
SUBTYPE_LABEL_WRAP = {
    "MC_CD14_S100A8_CD163_RETN": "CD14 S100A8\nCD163 RETN",
    "MC_NLRP3_HK2": "NLRP3 HK2",
    "MC_CD16_CX3CR1": "CD16 CX3CR1",
    "MC_SPIB_HLA-DRA_FLT3": "SPIB HLA-DRA\nFLT3",
}
SUBTYPE_SHORT = {
    "MC_CD14_S100A8_CD163_RETN": "CD14 S100A8",
    "MC_NLRP3_HK2": "NLRP3 HK2",
    "MC_CD16_CX3CR1": "CD16 CX3CR1",
    "MC_SPIB_HLA-DRA_FLT3": "SPIB HLA-DRA",
}
SUBTYPE_TINY = {
    "MC_CD14_S100A8_CD163_RETN": "CD14",
    "MC_NLRP3_HK2": "NLRP3",
    "MC_CD16_CX3CR1": "CD16",
    "MC_SPIB_HLA-DRA_FLT3": "SPIB",
}
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER,
                          ["#4E79A7", "#C0392B", "#59A14F", "#B07AA1"]))

MARKERS = ["CD14", "S100A8", "CD163", "RETN", "NLRP3", "HK2", "FCGR3A",
           "CX3CR1", "SPIB", "HLA-DRA", "FLT3", "IL1B", "CASP1", "TNF", "KYNU"]
EFFECTORS = ["NLRP3", "HK2", "IL1B", "CASP1", "KYNU"]

CD8_ORDER = ["CCR7_TCF7_IL7R", "TCF7_IL7R", "KLRB1_CCR7_IL7R", "MHC", "GZMH",
             "GZMH_NKG7", "GZMK_TIGIT_HAVCR2", "MKI67"]
CD8_COLORS = dict(zip(CD8_ORDER, [
    "#8DA0CB", "#66C2A5", "#A6D854", "#E5C494", "#FFD92F", "#E28E2C",
    "#C0392B", "#9C755F",
]))
CD8_LABEL = {c: c.replace("_", " ") for c in CD8_ORDER}

FERRITIN_COLS = {
    "CD14_S100A8": "MC_CD14_S100A8_CD163_RETN",
    "NLRP3_HK2": "MC_NLRP3_HK2",
    "CD16_CX3CR1": "MC_CD16_CX3CR1",
    "SPIB_HLA-DRA": "MC_SPIB_HLA-DRA_FLT3",
}

NOTES: list[str] = []


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    return np.nan_to_num((mat - mat.mean(axis=1, keepdims=True))
                         / (mat.std(axis=1, keepdims=True) + 1e-9))


def hvg_matrix(df: pd.DataFrame, col: str, order: list[str]):
    """Order the HVG panel up-genes first, then z-score each gene across `order`."""
    wide = df.pivot_table(index="gene", columns=col, values="mean_expr")
    reg = df.drop_duplicates("gene").set_index("gene")["regulation"]
    genes = (reg.to_frame().assign(rank=lambda d: d["regulation"].map({"Up": 0, "Down": 1}))
             .sort_values("rank").index.tolist())
    wide = wide.reindex(genes)[order]
    return zscore_rows(wide.to_numpy()), genes, reg.reindex(genes).tolist()


def main() -> None:
    apply_publication_style()

    cells = pd.read_csv(DATA / "fig8_cells.csv")
    prop = pd.read_csv(DATA / "fig8_patient_proportions.csv")
    hvg_sub = pd.read_csv(DATA / "fig8_hvg_by_subtype.csv")
    hvg_grp = pd.read_csv(DATA / "fig8_hvg_by_group.csv")
    chat = pd.read_csv(DATA / "fig8_cellchat.csv")
    ferr = pd.read_csv(DATA / "fig8_ferritin.csv")

    NOTES.append(f"panels a-g: {len(cells):,} monocytes over {GROUP_ORDER} "
                 "(IM_M and MH_CD56 excluded to keep the four-group continuum)")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.55))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.52, 1.02, 1.72, 1.86, 1.44],
                             left=0.068, right=0.962, top=0.972, bottom=0.045,
                             hspace=0.52)

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.34, 0.016, 0.17],
                                wspace=0.10)
    ax_a = fig.add_subplot(row1[0])
    P.umap_by_category(ax_a, cells, "celltype", SUBTYPE_COLORS,
                       SUBTYPE_LABEL_WRAP, size=0.5, min_dist_frac=1.05)
    ax_a.set_title(f"{len(cells):,} monocytes  |  4 states", fontsize=5.2,
                   color=NEUTRAL_MID, pad=1.5)

    ax_b = fig.add_subplot(row1[1])
    cax_b = fig.add_subplot(row1[2])
    side_b = row1[3].subgridspec(2, 1, hspace=0.25)
    lax_b = fig.add_subplot(side_b[1])
    P.dotplot(fig, ax_b, cax_b, lax_b, cells, MARKERS, "celltype", SUBTYPE_ORDER,
              SUBTYPE_LABEL, genes_on_x=True, size_scale=0.32)

    # ---- row 2: c | d -----------------------------------------------------
    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.05, 0.16, 1.25], wspace=0.10)
    grid_c = row2[0].subgridspec(1, len(GROUP_ORDER), wspace=0.06)
    axes_c = []
    for i, g in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(grid_c[i])
        axes_c.append(ax)
        ax.scatter(cells["UMAP1"], cells["UMAP2"], s=0.35, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        sub = cells[cells["group"] == g]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.4, linewidths=0, alpha=0.85,
                   color=[SUBTYPE_COLORS[c] for c in sub["celltype"]],
                   rasterized=True)
        strip_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.2, pad=1.2)
    NOTES.append("panel c: grey points show all monocytes as context; coloured "
                 "points are that group only, keeping the state palette of a")

    grid_d = row2[2].subgridspec(1, 4, wspace=0.78)
    axes_d = [fig.add_subplot(grid_d[i]) for i in range(4)]
    NOTES.extend(P.proportion_boxplots(axes_d, prop, SUBTYPE_ORDER, SUBTYPE_SHORT,
                                       ncol=4))

    # ---- row 3: e | f | g -------------------------------------------------
    row3 = outer[2].subgridspec(1, 7,
                                width_ratios=[1.0, 0.022, 0.42, 1.0, 0.022, 0.44,
                                              0.66],
                                wspace=0.14)
    reg_colors = {"Up": "#C0392B", "Down": "#4E79A7"}

    z_e, genes_e, reg_e = hvg_matrix(hvg_sub, "subtype", SUBTYPE_ORDER)
    cax_e = fig.add_subplot(row3[1])
    anchor_e = P.annotated_block_heatmap(
        fig, row3[0], cax_e, z_e, genes_e,
        [SUBTYPE_TINY[s] for s in SUBTYPE_ORDER], reg_e, reg_colors,
        nblocks=2, wspace=1.35, col_rotation=90)
    anchor_e.set_title("Genes separating NLRP3 HK2", fontsize=5.2, pad=3.0,
                       loc="left")

    z_f, genes_f, reg_f = hvg_matrix(hvg_grp, "group", GROUP_ORDER)
    cax_f = fig.add_subplot(row3[4])
    anchor_f = P.annotated_block_heatmap(
        fig, row3[3], cax_f, z_f, genes_f,
        [GROUP_LABELS[g] for g in GROUP_ORDER], reg_f, reg_colors,
        nblocks=2, wspace=1.35, col_rotation=90)
    anchor_f.set_title("The same genes within NLRP3 HK2", fontsize=5.2, pad=3.0,
                       loc="left")
    NOTES.append(f"panels e,f: {len(genes_e)} genes, the 10 most up- and 10 most "
                 "down-regulated among the 2,000 most variable genes of "
                 "MC_NLRP3_HK2 (ribosomal, mitochondrial and unnamed loci "
                 "excluded); red strip = up in NLRP3 HK2, blue = down; each gene "
                 "z-scored across the columns of its own panel")

    grid_g = row3[6].subgridspec(len(EFFECTORS), 1, hspace=0.14)
    axes_g = [fig.add_subplot(grid_g[i]) for i in range(len(EFFECTORS))]
    P.stacked_violin(axes_g, cells, EFFECTORS, "celltype", SUBTYPE_ORDER,
                     SUBTYPE_COLORS, SUBTYPE_TINY)
    axes_g[0].set_title("Inflammasome and\ntryptophan effectors", fontsize=5.2,
                        pad=2.0)

    # ---- row 4: h | i -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 7,
                                width_ratios=[0.30, 0.78, 0.020, 0.52, 1.28,
                                              0.020, 0.40],
                                wspace=0.14)
    totals = (chat.groupby(["Newgroup6", "CD8_subtype"])["prob"].sum()
              .unstack(fill_value=0.0).reindex(index=GROUP_ORDER, fill_value=0.0))
    for c in CD8_ORDER:
        if c not in totals.columns:
            totals[c] = 0.0
    totals = totals[CD8_ORDER]
    z_h = zscore_rows(totals.to_numpy().T)
    ax_h = fig.add_subplot(row4[1])
    cax_h = fig.add_subplot(row4[2])
    vmax_h = float(np.abs(z_h).max())
    im_h = ax_h.imshow(z_h, cmap=DIVERGING_CMAP, aspect="auto", vmin=-vmax_h,
                       vmax=vmax_h)
    ax_h.set_xticks(range(len(GROUP_ORDER)))
    ax_h.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=40,
                         ha="right", rotation_mode="anchor")
    ax_h.set_yticks(range(len(CD8_ORDER)))
    ax_h.set_yticklabels([CD8_LABEL[c] for c in CD8_ORDER], fontsize=5.0)
    ax_h.tick_params(length=0, pad=1.0)
    for spine in ax_h.spines.values():
        spine.set_visible(False)
    for i, c in enumerate(CD8_ORDER):
        ax_h.add_patch(plt.Rectangle((-0.5, i - 0.5), len(GROUP_ORDER), 1,
                                     fill=False, edgecolor="white", linewidth=0.4))
    cb_h = fig.colorbar(im_h, cax=cax_h)
    cb_h.ax.set_title("z-score", fontsize=5.0, pad=2.0)
    cb_h.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb_h.outline.set_linewidth(0.4)
    ax_h.set_title("Signalling strength to CD8", fontsize=5.2, pad=3.0, loc="left")

    # The pairs worth showing are the ones that move: rank by variance across the
    # four groups, after dropping pairs that are weak everywhere. Probability is
    # summed over the CD8 subtypes, which panel h already resolves. MHC-I alone
    # supplies the eight strongest movers, so each pathway is capped at three
    # pairs to keep the panel from becoming a list of HLA genes.
    lr = (chat.pivot_table(index=["pathway_name", "interaction_name_2"],
                           columns="Newgroup6", values="prob", aggfunc="sum",
                           fill_value=0.0)
          .reindex(columns=GROUP_ORDER, fill_value=0.0).reset_index())
    vals = lr[GROUP_ORDER].to_numpy()
    lr["max_prob"], lr["var_prob"] = vals.max(axis=1), vals.var(axis=1, ddof=1)
    lr = lr[lr["max_prob"] >= np.quantile(lr["max_prob"], 0.5)]
    lr = lr.sort_values("var_prob", ascending=False)
    path_rank = (lr.groupby("pathway_name")["var_prob"].max()
                 .sort_values(ascending=False).head(6).index.tolist())
    lr = lr[lr["pathway_name"].isin(path_rank)].groupby("pathway_name",
                                                        sort=False).head(3)
    lr["pathway_name"] = pd.Categorical(lr["pathway_name"], path_rank)
    lr = lr.sort_values(["pathway_name", "var_prob"], ascending=[True, False])
    path_colors = dict(zip(path_rank, [
        "#C0392B", "#4E79A7", "#59A14F", "#E28E2C", "#B07AA1", "#76B7B2",
    ]))
    z_i = zscore_rows(lr[GROUP_ORDER].to_numpy())
    cax_i = fig.add_subplot(row4[5])
    anchor_i = P.annotated_block_heatmap(
        fig, row4[4], cax_i, z_i, lr["interaction_name_2"].tolist(),
        [GROUP_LABELS[g] for g in GROUP_ORDER],
        lr["pathway_name"].astype(str).tolist(), path_colors, nblocks=2,
        wspace=1.05, italic=False, col_rotation=90)
    anchor_i.set_title("Ligand-receptor pairs that move across the continuum",
                       fontsize=5.2, pad=3.0, loc="left")
    lax_i = fig.add_subplot(row4[6])
    strip_axes(lax_i)
    leg_i = lax_i.legend(
        handles=[Line2D([], [], marker="s", linestyle="none", markersize=2.6,
                        markerfacecolor=path_colors[p], markeredgecolor="none",
                        label=p) for p in path_rank],
        loc="center left", fontsize=5.0, handletextpad=0.35, labelspacing=0.30,
        borderpad=0.1, borderaxespad=0.0, title="Pathway")
    leg_i.get_title().set_fontsize(5.0)
    NOTES.append(f"panels h-j: CellChat interactions from MC_NLRP3_HK2 to "
                 f"{len(CD8_ORDER)} CD8 subtypes, {len(chat):,} significant "
                 f"source-target-pair records; panel i shows {len(lr)} pairs from "
                 f"the {len(path_rank)} pathways with the largest between-group "
                 "variance, at most three pairs each, drawn from those in the "
                 "upper half of peak probability, summed over CD8 subtypes and "
                 "z-scored per row")

    # ---- row 5: j | k -----------------------------------------------------
    row5 = outer[4].subgridspec(1, 4, width_ratios=[0.86, 0.30, 0.10, 1.90],
                                wspace=0.16)
    ax_j = fig.add_subplot(row5[0])
    x = np.arange(len(GROUP_ORDER))
    for c in CD8_ORDER:
        ax_j.plot(x, totals[c].to_numpy(), color=CD8_COLORS[c], linewidth=0.7,
                  marker="o", markersize=1.6, markeredgewidth=0)
    ax_j.set_xticks(x)
    ax_j.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=40,
                         ha="right", rotation_mode="anchor")
    ax_j.set_ylabel("Total interaction probability", labelpad=1.5)
    ax_j.set_xlim(-0.35, len(GROUP_ORDER) - 0.65)
    ax_j.tick_params(length=1.2, pad=1.0)
    ax_j.grid(True, axis="y", color="#EDEDED", linewidth=0.3)
    ax_j.set_axisbelow(True)
    lax_j = fig.add_subplot(row5[1])
    strip_axes(lax_j)
    leg_j = lax_j.legend(
        handles=[Line2D([], [], color=CD8_COLORS[c], linewidth=0.9,
                        label=CD8_LABEL[c]) for c in CD8_ORDER],
        loc="center left", fontsize=5.0, handlelength=1.2, handletextpad=0.35,
        labelspacing=0.30, borderpad=0.1, borderaxespad=0.0, title="CD8 subtype")
    leg_j.get_title().set_fontsize(5.0)

    grid_k = row5[3].subgridspec(1, 4, wspace=0.30)
    axes_k = []
    rng = np.random.default_rng(7)
    for i, (col, subtype) in enumerate(FERRITIN_COLS.items()):
        ax = fig.add_subplot(grid_k[i])
        axes_k.append(ax)
        sub = ferr[["Newgroup6", col, "ferritin"]].dropna()
        sub = sub[sub["ferritin"] > 0]
        xv = sub[col].to_numpy(dtype=float)
        yv = np.log10(sub["ferritin"].to_numpy(dtype=float))
        for g in ("MH_CD4", "HLH"):
            m = (sub["Newgroup6"] == g).to_numpy()
            ax.scatter(xv[m], yv[m], s=4.0, linewidths=0.15, edgecolors="white",
                       color=GROUP_COLORS[g], zorder=3)
        if len(xv) >= 4:
            rho, pval = stats.spearmanr(xv, yv)
            slope, icept = np.polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs, slope * xs + icept, color=NEUTRAL_DARK, linewidth=0.5,
                    zorder=2)
            ax.set_title(f"{col.replace('_', ' ')}\n"
                         f"rho = {rho:.2f}, P = {pval:.2g}", fontsize=5.0, pad=2.0)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["10", "100", "1,000", "10,000"])
        ax.set_xlabel("Proportion (%)", labelpad=1.0)
        ax.tick_params(length=1.2, pad=1.0)
        ax.margins(0.12)
        if i == 0:
            ax.set_ylabel("Serum ferritin (ng/mL)", labelpad=1.5)
    leg_k = axes_k[-1].legend(
        handles=[Line2D([], [], marker="o", linestyle="none", markersize=2.2,
                        markerfacecolor=GROUP_COLORS[g], markeredgecolor="none",
                        label=GROUP_LABELS[g]) for g in ("MH_CD4", "HLH")],
        loc="upper right", fontsize=5.0, handletextpad=0.35, labelspacing=0.25,
        borderpad=0.2, frameon=False)
    NOTES.append(f"panel k: {len(ferr)} participants with a paired ferritin "
                 "measurement (HLH and MH-CD4 only; HC and IM were not assayed); "
                 "Spearman correlation on the untransformed values, line is an "
                 "ordinary least-squares fit on log10 ferritin")

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (axes_c[0], "c"), (axes_d[0], "d"),
                    (anchor_e, "e"), (anchor_f, "f"), (axes_g[0], "g"),
                    (ax_h, "h"), (anchor_i, "i"), (ax_j, "j"), (axes_k[0], "k")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure8")
    (OUT / "Figure8_provenance.txt").write_text(
        "Figure 8 | An NLRP3/HK2 inflammatory monocyte state expands with "
        "disease severity and rewires monocyte-to-CD8 signalling\nRendered in "
        "Python (matplotlib); R used only to export data from the Seurat "
        "object.\n\n" + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
