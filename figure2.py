"""Figure 2 | CD8+ T cell compartment remodelling across the EBV continuum.

Core conclusion: the CD8+ compartment shifts from naive/memory-dominant in health
towards cytotoxic, proliferating and exhaustion-marked subsets as disease severity
increases, and this shift is mirrored by a coordinated transcriptional programme.

Panels
  a  UMAP of 95,863 CD8+ T cells resolved into 8 subtypes
  b  Subtype-defining marker expression
  c  The same embedding split by disease group
  d  Discriminative marker panel per subtype
  e  Per-patient subtype proportions across groups
  f  Group-level expression of transcription-factor, cytokine and surface-marker modules
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
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig2"
OUT = ROOT / "Figure2"

SUBTYPE_ORDER = [
    "CD8T_CCR7_TCF7_IL7R", "CD8T_TCF7_IL7R", "CD8T_KLRB1_CCR7_IL7R",
    "CD8T_GZMH", "CD8T_GZMH_NKG7", "CD8T_GZMK_TIGIT_HAVCR2",
    "CD8T_MKI67", "CD8T_MHC",
]
SUBTYPE_LABEL = {s: s.replace("CD8T_", "").replace("_", " ") for s in SUBTYPE_ORDER}
# Two-line variants keep the three-marker names compact enough to sit on the embedding.
SUBTYPE_LABEL_WRAP = dict(SUBTYPE_LABEL, **{
    "CD8T_CCR7_TCF7_IL7R": "CCR7 TCF7\nIL7R",
    "CD8T_KLRB1_CCR7_IL7R": "KLRB1 CCR7\nIL7R",
    "CD8T_GZMK_TIGIT_HAVCR2": "GZMK TIGIT\nHAVCR2",
})
SUBTYPE_COLORS = dict(zip(SUBTYPE_ORDER, [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]))

VIOLIN_GENES = ["GZMH", "CCR7", "TYMS", "PCLAF", "IL7R", "GZMK", "LEF1", "MKI67", "TIGIT"]
BUBBLE_GENES = [
    "LEF1-AS1", "CCR7", "SCML1", "ADGRG1", "PPP2R2B", "FCGR3A", "LAIR2", "FGFBP2",
    "CXCR6", "TIMD4", "GZMK", "CSGALNACT1", "PLCL1", "IGF1R", "GINS2", "MCM2",
    "E2F1", "HJURP", "UBE2C", "SPC25", "RTKN2", "LSR", "IFNG-AS1",
]
MODULES = {
    "Transcription factor": ["TCF7", "LEF1", "ZEB1", "ID3", "PRDM1", "TBX21",
                             "BHLHE40", "RORC", "RORA", "ZEB2", "ID2", "TOX",
                             "TOX2", "BATF", "TRIB1"],
    "Cytokine": ["IFNG", "TNF", "GZMA", "GZMB", "GZMK", "GZMH", "GNLY", "PRF1"],
    "Surface marker": ["SELL", "CCR7", "CXCR3", "CD28", "CD69", "IL2RA", "TNFRSF4",
                       "TNFRSF9", "ICOS", "CD38", "KLRG1", "CX3CR1", "PDCD1",
                       "ENTPD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2"],
}
MODULE_COLORS = {"Transcription factor": "#7BAA5B", "Cytokine": "#C0392B",
                 "Surface marker": "#4C7FB8"}

NOTES: list[str] = []


def main() -> None:
    apply_publication_style()
    cells = pd.read_csv(DATA / "fig2_cells.csv")
    prop = pd.read_csv(DATA / "fig2_patient_proportions.csv")
    NOTES.append(f"panels a-d,f: {len(cells):,} CD8+ T cells, "
                 f"{cells['celltype'].nunique()} subtypes, groups {GROUP_ORDER} "
                 "(IM_M and MH_CD56 excluded to keep the four-group continuum)")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.30))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.95, 1.42, 1.70, 3.10],
                             left=0.075, right=0.955, top=0.975, bottom=0.058,
                             hspace=0.38)

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.32], wspace=0.14)
    ax_a = fig.add_subplot(row1[0])
    P.umap_by_category(ax_a, cells, "celltype", SUBTYPE_COLORS, SUBTYPE_LABEL_WRAP,
                       min_dist_frac=0.44)
    ax_a.set_title(f"{len(cells):,} CD8+ T cells  |  8 subtypes", fontsize=5.2,
                   color=NEUTRAL_MID, pad=1.5)

    grid_b = row1[1].subgridspec(len(VIOLIN_GENES), 1, hspace=0.0)
    axes_b = [fig.add_subplot(grid_b[i]) for i in range(len(VIOLIN_GENES))]
    P.stacked_violin(axes_b, cells, VIOLIN_GENES, "celltype", SUBTYPE_ORDER,
                     SUBTYPE_COLORS, SUBTYPE_LABEL)

    # ---- row 2: c (embedding split by disease group) ----------------------
    grid_c = outer[1].subgridspec(1, len(GROUP_ORDER), wspace=0.05)
    axes_c = [fig.add_subplot(grid_c[i]) for i in range(len(GROUP_ORDER))]
    for ax, g in zip(axes_c, GROUP_ORDER):
        sub = cells[cells["group"] == g]
        ax.scatter(cells["UMAP1"], cells["UMAP2"], s=0.25, linewidths=0,
                   color="#E8E8E8", rasterized=True)
        for ct in SUBTYPE_ORDER:
            part = sub[sub["celltype"] == ct]
            ax.scatter(part["UMAP1"], part["UMAP2"], s=0.35, linewidths=0,
                       color=SUBTYPE_COLORS[ct], alpha=0.8, rasterized=True)
        from nature_style import embedding_axes

        embedding_axes(ax)
        ax.set_title(f"{GROUP_LABELS[g]}  (n = {len(sub):,})", fontsize=5.2, pad=1.2)
    NOTES.append("panel c: grey points show all cells as context; coloured points are "
                 "the cells of that group only")

    # ---- row 3: d (discriminative marker dot plot) ------------------------
    # The leading spacer gives the long subtype names room outside the plot area.
    row3 = outer[2].subgridspec(1, 4, width_ratios=[0.105, 1.0, 0.014, 0.155],
                               wspace=0.035)
    ax_d = fig.add_subplot(row3[1])
    cax_d = fig.add_subplot(row3[2])
    side_d = row3[3].subgridspec(2, 1, hspace=0.25)
    lax_d = fig.add_subplot(side_d[1])
    P.dotplot(fig, ax_d, cax_d, lax_d, cells, BUBBLE_GENES, "celltype",
              SUBTYPE_ORDER, SUBTYPE_LABEL, genes_on_x=True, size_scale=0.30)

    # ---- row 4: e | f -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 4, width_ratios=[1.0, 0.13, 0.52, 0.028],
                               wspace=0.14)
    grid_e = row4[0].subgridspec(2, 4, wspace=0.50, hspace=1.00)
    axes_e = [fig.add_subplot(grid_e[i, j]) for i in range(2) for j in range(4)]
    NOTES.extend(P.proportion_boxplots(axes_e, prop, SUBTYPE_ORDER, SUBTYPE_LABEL,
                                       ncol=4))

    cax_f = fig.add_subplot(row4[3])
    genes_f, ax_f_anchor = P.module_heatmap(fig, row4[2], cax_f, cells, MODULES,
                                            module_colors=MODULE_COLORS, nblocks=2)
    NOTES.append(f"panel f: {len(genes_f)} genes present of "
                 f"{len(set(sum(MODULES.values(), [])))} requested; group-level mean "
                 "expression z-scored across the four groups")

    handles = [Line2D([], [], marker="s", linestyle="none", markersize=3,
                      markerfacecolor=MODULE_COLORS[m], markeredgecolor="none", label=m)
               for m in MODULES]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.955, 0.002),
               ncol=3, fontsize=5.0, handletextpad=0.4, columnspacing=1.0)

    for ax, lab in ((ax_a, "a"), (axes_b[0], "b"), (axes_c[0], "c"), (ax_d, "d"),
                    (axes_e[0], "e"), (ax_f_anchor, "f")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure2")
    (OUT / "Figure2_provenance.txt").write_text(
        "Figure 2 | CD8+ T cell compartment remodelling across the EBV continuum\n"
        "Rendered in Python (matplotlib); R used only to export data from the Seurat object.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
