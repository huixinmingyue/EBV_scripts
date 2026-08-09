"""Figure 9 | The CD4 compartment shifts from naive-memory to effector states
and its regulatory arm loses stability with disease severity.

Core conclusion: CD4 T cells reorganise along the same continuum as the CD8
compartment. Naive CCR7/LEF1 cells give way to IL7R memory, Th1 and LAYN
regulatory states; the MH-CD4 group carries a strongly oxidative,
antigen-presenting programme relative to IM; and within the Treg pool the
homeostatic signature falls while the progressive one rises, alongside a shift
from the FCER1G to the LAYN regulatory subset.

Panels
  a  CD4 embedding resolved into eight subtypes
  b  Subtype-defining marker expression
  c  The same embedding split by group
  d  Treg homeostatic and progressive module scores across groups
  e  Per-patient subtype proportions across groups
  f  CD4 functional module scores, MH-CD4 against IM
  g  KEGG pathways separating MH-CD4 from IM
  h  Treg signature gene expression across groups
  i  The same genes across the CD4 subtypes
  j  Regulatory subset proportions among CD4 T cells
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import panels as P
from nature_style import (
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig9"
FIG09 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig09")
OUT = ROOT / "Figure9"

SUBTYPE_ORDER = ["CD4_CCR7", "CD4_LEF1", "CD4_IL7R", "CD4_CXCR5_BCL6",
                 "CD4_CXCR5", "Th1", "Treg_LAYN", "Treg_FCER1G"]
SUBTYPE_LABEL = {
    "CD4_CCR7": "CCR7", "CD4_LEF1": "LEF1", "CD4_IL7R": "IL7R",
    "CD4_CXCR5_BCL6": "CXCR5 BCL6", "CD4_CXCR5": "CXCR5", "Th1": "Th1",
    "Treg_LAYN": "Treg LAYN", "Treg_FCER1G": "Treg FCER1G",
}
SUBTYPE_LABEL_WRAP = dict(SUBTYPE_LABEL, **{
    "CD4_CXCR5_BCL6": "CXCR5\nBCL6", "Treg_LAYN": "Treg\nLAYN",
    "Treg_FCER1G": "Treg\nFCER1G",
})
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER, [
    "#4E79A7", "#8DA0CB", "#76B7B2", "#59A14F", "#A6D854", "#E28E2C",
    "#C0392B", "#B07AA1",
]))

MARKERS = ["CCR7", "LEF1", "SELL", "TCF7", "IL7R", "CXCR5", "BCL6", "ICOS",
           "IFNG", "TBX21", "GZMB", "LAYN", "TNFRSF9", "FCER1G", "FOXP3",
           "CTLA4", "IL2RA", "IKZF2"]
TREG_GENES = ["FOXP3", "CTLA4", "TGFBR1", "IL2RA", "IKZF2"]

# The remaining module columns are all-NaN upstream: the signatures did not
# score on this object, so they are dropped rather than drawn as empty facets.
CD4_MODULES = ["Memory_score", "Exhaustion_score", "Cytotoxicity_score",
               "Proliferation_score", "Th1_score", "Treg_function_score"]
MODULE_LABEL = {m: m.replace("_score", "").replace("_", " ") for m in CD4_MODULES}
MODULE_LABEL["Treg_function_score"] = "Treg fn."

HP_MODULES = ["Homeostatic_score", "Progressive_score"]
TREG_PROPS = ["prop_Treg", "prop_Treg_FCER1G", "prop_Treg_LAYN"]
PROP_LABEL = {"prop_Treg": "All Treg", "prop_Treg_FCER1G": "Treg FCER1G",
              "prop_Treg_LAYN": "Treg LAYN"}

NOTES: list[str] = []


def kruskal_p(df: pd.DataFrame, col: str, group_col: str = "Newgroup6") -> float:
    arms = [df.loc[df[group_col] == g, col].dropna().to_numpy() for g in GROUP_ORDER]
    arms = [a for a in arms if len(a) > 0]
    return stats.kruskal(*arms).pvalue if len(arms) >= 2 else np.nan


def main() -> None:
    apply_publication_style()

    cells = pd.read_csv(DATA / "fig9_cells.csv")
    prop = pd.read_csv(DATA / "fig9_patient_proportions.csv").rename(
        columns={"orig.ident": "sample"})
    modules = pd.read_csv(FIG09 / "P01_CD4_MHCD4_vs_IM" / "module_scores_per_sample.csv")
    mod_stats = pd.read_csv(FIG09 / "P01_CD4_MHCD4_vs_IM" / "module_scores_wilcox.csv")
    gsea = pd.read_csv(FIG09 / "P01_CD4_MHCD4_vs_IM" / "mhcd4_vs_im_GSEA_KEGG.csv")
    hp = pd.read_csv(FIG09 / "P03_Treg_function" / "treg_HP_module_per_sample.csv")
    treg_expr = pd.read_csv(FIG09 / "P03_Treg_function" / "treg_key_gene_means_per_sample.csv")
    treg_prop = pd.read_csv(FIG09 / "P03_Treg_function" / "treg_proportion_by_sample.csv")

    NOTES.append(f"panels a-c,e,i: {len(cells):,} CD4 T cells over {GROUP_ORDER} "
                 "(IM_M and MH_CD56 excluded to keep the four-group continuum); "
                 "embedding is the harmony-integrated UMAP")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 8.70))
    outer = fig.add_gridspec(5, 1, height_ratios=[1.50, 1.05, 0.98, 1.58, 1.42],
                             left=0.068, right=0.962, top=0.972, bottom=0.048,
                             hspace=0.56)

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.42, 0.016, 0.17],
                                wspace=0.10)
    ax_a = fig.add_subplot(row1[0])
    P.umap_by_category(ax_a, cells, "celltype", SUBTYPE_COLORS,
                       SUBTYPE_LABEL_WRAP, size=0.35, min_dist_frac=1.25)
    ax_a.set_title(f"{len(cells):,} CD4 T cells  |  8 subtypes", fontsize=5.2,
                   color=NEUTRAL_MID, pad=1.5)

    ax_b = fig.add_subplot(row1[1])
    cax_b = fig.add_subplot(row1[2])
    side_b = row1[3].subgridspec(2, 1, hspace=0.25)
    lax_b = fig.add_subplot(side_b[1])
    P.dotplot(fig, ax_b, cax_b, lax_b, cells, MARKERS, "celltype", SUBTYPE_ORDER,
              SUBTYPE_LABEL, genes_on_x=True, size_scale=0.30)

    # ---- row 2: c | d -----------------------------------------------------
    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.55, 0.22, 0.72], wspace=0.10)
    grid_c = row2[0].subgridspec(1, len(GROUP_ORDER), wspace=0.06)
    axes_c = []
    for i, g in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(grid_c[i])
        axes_c.append(ax)
        ax.scatter(cells["UMAP1"], cells["UMAP2"], s=0.25, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        sub = cells[cells["group"] == g]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.30, linewidths=0, alpha=0.85,
                   color=[SUBTYPE_COLORS[c] for c in sub["celltype"]],
                   rasterized=True)
        strip_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.2, pad=1.2)
    NOTES.append("panel c: grey points show all CD4 T cells as context; coloured "
                 "points are that group only, keeping the subtype palette of a")

    grid_d = row2[2].subgridspec(1, 2, wspace=0.62)
    axes_d = [fig.add_subplot(grid_d[i]) for i in range(2)]
    P.group_boxplots(axes_d, hp, HP_MODULES, "Newgroup6", GROUP_ORDER,
                     labels={m: m.replace("_score", "") for m in HP_MODULES},
                     pvalues={m: kruskal_p(hp, m) for m in HP_MODULES},
                     ylabel="Module score")
    NOTES.append(f"panel d: per-patient Treg module scores over {len(hp)} samples; "
                 "the homeostatic and progressive CD4 signatures are the same two "
                 "gene sets used for the CD8 axes in Figure 3")

    # ---- row 3: e ---------------------------------------------------------
    grid_e = outer[2].subgridspec(1, 8, wspace=0.62)
    axes_e = [fig.add_subplot(grid_e[i]) for i in range(8)]
    NOTES.extend(P.proportion_boxplots(axes_e, prop, SUBTYPE_ORDER, SUBTYPE_LABEL,
                                       ncol=8))

    # ---- row 4: f | g -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 3, width_ratios=[1.14, 0.34, 1.12], wspace=0.10)
    grid_f = row4[0].subgridspec(2, 3, wspace=0.66, hspace=0.78)
    axes_f = [fig.add_subplot(grid_f[i // 3, i % 3]) for i in range(len(CD4_MODULES))]
    fdr = mod_stats.set_index("module")["p_fdr"].to_dict()
    P.group_boxplots(axes_f, modules, CD4_MODULES, "Newgroup6", ["IM", "MH_CD4"],
                     labels=MODULE_LABEL, pvalues=fdr, ylabel="Module score")
    axes_f[3].set_ylabel("Module score", fontsize=5.0, labelpad=1.5)
    NOTES.append("panel f: per-patient CD4 module scores, IM against MH-CD4 "
                 f"({(modules['Newgroup6'] == 'IM').sum()} and "
                 f"{(modules['Newgroup6'] == 'MH_CD4').sum()} samples); "
                 "Wilcoxon rank-sum with Benjamini-Hochberg correction across the "
                 "modules, as computed upstream; four further signatures scored "
                 "all-NaN on this object and are not shown")

    ax_g = fig.add_subplot(row4[2])
    sig = gsea[gsea["padj"] < 0.05].copy()
    P.nes_barplot(ax_g, sig, "Description", "NES", GROUP_COLORS["MH_CD4"],
                  GROUP_COLORS["IM"], xlabel="NES (MH-CD4 vs IM)",
                  max_label_chars=42)
    ax_g.set_title(f"{len(sig)} KEGG pathways, FDR < 0.05", fontsize=5.2, pad=2.0,
                   loc="left")
    NOTES.append("panel g: fgsea over KEGG with genes ranked by the MH-CD4 vs IM "
                 "differential test; positive NES is enriched in MH-CD4")

    # ---- row 5: h | i | j -------------------------------------------------
    row5 = outer[4].subgridspec(1, 6,
                                width_ratios=[1.06, 0.30, 0.18, 0.94, 0.20, 0.76],
                                wspace=0.10)
    ax_h = fig.add_subplot(row5[0])
    # The five genes sit at very different absolute levels, so a shared raw axis
    # would flatten TGFBR1 against IKZF2. Each gene is centred across patients,
    # which is what the panel is actually asking about.
    treg_z = treg_expr.copy()
    treg_z[TREG_GENES] = ((treg_z[TREG_GENES] - treg_z[TREG_GENES].mean())
                          / treg_z[TREG_GENES].std())
    P.grouped_boxplots(ax_h, treg_z, TREG_GENES, "Newgroup6", GROUP_ORDER,
                       pvalues={g: kruskal_p(treg_expr, g) for g in TREG_GENES},
                       ylabel="Mean expression (z across patients)", italic=True)
    ax_h.set_title("Treg signature across groups", fontsize=5.2, pad=2.0, loc="left")
    lax_h = fig.add_subplot(row5[1])
    strip_axes(lax_h)
    lax_h.legend(handles=[Patch(facecolor=GROUP_COLORS[g], edgecolor="none",
                                label=GROUP_LABELS[g]) for g in GROUP_ORDER],
                 loc="center left", fontsize=5.0, handlelength=1.0,
                 handletextpad=0.35, labelspacing=0.30, borderpad=0.1,
                 borderaxespad=0.0)
    NOTES.append(f"panel h: per-patient mean expression within the Treg pool over "
                 f"{len(treg_expr)} samples, z-scored per gene across patients; "
                 "Kruskal-Wallis across the four groups on the untransformed means")

    grid_i = row5[3].subgridspec(len(TREG_GENES), 1, hspace=0.14)
    axes_i = [fig.add_subplot(grid_i[i]) for i in range(len(TREG_GENES))]
    P.stacked_violin(axes_i, cells, TREG_GENES, "celltype", SUBTYPE_ORDER,
                     SUBTYPE_COLORS, SUBTYPE_LABEL)
    axes_i[0].set_title("Treg signature by subtype", fontsize=5.2, pad=2.0)

    ax_j = fig.add_subplot(row5[5])
    treg_prop_pct = treg_prop.copy()
    treg_prop_pct[TREG_PROPS] = treg_prop_pct[TREG_PROPS] * 100
    P.grouped_boxplots(ax_j, treg_prop_pct, TREG_PROPS, "Newgroup6", GROUP_ORDER,
                       labels=PROP_LABEL,
                       pvalues={c: kruskal_p(treg_prop_pct, c) for c in TREG_PROPS},
                       ylabel="Proportion of CD4 (%)")
    ax_j.set_title("Regulatory subsets", fontsize=5.2, pad=2.0, loc="left")
    NOTES.append("panel j: regulatory subsets as a per-cent of that patient's CD4 "
                 "T cells; Kruskal-Wallis across the four groups")

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (axes_c[0], "c"), (axes_d[0], "d"),
                    (axes_e[0], "e"), (axes_f[0], "f"), (ax_g, "g"),
                    (ax_h, "h"), (axes_i[0], "i"), (ax_j, "j")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure9")
    (OUT / "Figure9_provenance.txt").write_text(
        "Figure 9 | The CD4 compartment shifts from naive-memory to effector "
        "states and its regulatory arm loses stability\nRendered in Python "
        "(matplotlib); R used only to export data from the Seurat object.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
