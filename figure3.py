"""Figure 3 | Opposing recovery and exhaustion programmes in CD8+ T cells.

Core conclusion: the acute-to-chronic step (MH-CD4 vs IM) and the chronic-to-HLH
step engage partly opposed transcriptional programmes; a homeostatic/progressive
axis tracks this and a set of pathways reverses direction between the two steps.

Panels
  a  Functional module scores, MH-CD4 vs IM
  b  Functional module scores, HLH vs MH-CD4
  c  KEGG GSEA, MH-CD4 vs IM
  d  KEGG GSEA, HLH vs MH-CD4
  e  Size of each differential gene set and their intersections
  f  Per-sample homeostatic/progressive ratio across the continuum
  g  KEGG pathways that reverse direction between the two axes
  h  Monotonic KEGG pathway trajectories, IM to HLH
  i  Genes with monotonic trends, group-mean expression
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
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
    significance_stars,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
REGEN = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out")
DATA = ROOT / "_data" / "fig3"
OUT = ROOT / "Figure3"

T01 = REGEN / "Fig02" / "TC_EBV_01_mhcd4_vs_im"
T02 = REGEN / "Fig02" / "TC_EBV_02_hlh_vs_mhcd4"
T03 = REGEN / "Fig03" / "TC_EBV_03_homeostatic_progressive"
T05 = REGEN / "Fig03" / "TC_EBV_05_monotonic_trend"

# The upstream analysis wrote nine module columns but only these five were ever
# populated; the other four are empty in the source tables.
MODULES = ["Cytotoxicity_score", "Inflammation_score", "Exhaustion_score",
           "Proliferation_score", "Memory_score"]
MODULE_LABEL = {m: m.replace("_score", "") for m in MODULES}

UP_COLOR = "#B02418"
DOWN_COLOR = "#2C6FAD"
SET_COLORS = ["#B02418", "#E28E2C", "#7BAA5B", "#2C6FAD", "#767676"]

INC_GENES = ["TRBV6-6", "TRAV30", "TRAV13-1", "SKAP1", "CCND3", "CX3CR1", "PPP2R2B",
             "S100A11", "HIVEP3", "MSC-AS1", "HLA-DQB1", "JAKMIP2", "SLC14A1", "CD4",
             "SH3BP5", "DPYD", "SP110", "KLF6", "TBC1D5", "RAP1GAP2"]
DEC_GENES = ["RBM38", "CREM", "DUSP2", "SAT1", "FTH1", "NR4A2", "LEPROTL1", "BANP",
             "CD27", "DDIT4", "DNAJB1", "HERPUD1", "CD8B", "HSPA8", "METRNL",
             "SNHG7", "MTFP1", "PER1", "ATP5MC2", "HSPA1B"]

NOTES: list[str] = []


def module_panel(fig, gs, groups, source, letter_axes):
    scores = pd.read_csv(source / "module_scores_per_sample.csv")
    wilcox = pd.read_csv(source / "module_scores_wilcox.csv")
    pvals = dict(zip(wilcox["module"], wilcox["p_wilcox"]))
    grid = gs.subgridspec(1, len(MODULES), wspace=0.62)
    axes = [fig.add_subplot(grid[i]) for i in range(len(MODULES))]
    P.group_boxplots(axes, scores, MODULES, "Newgroup6", groups, MODULE_LABEL,
                     pvalues=pvals, ylabel="Module score")
    letter_axes.append(axes[0])
    n = scores.groupby("Newgroup6").size().to_dict()
    return ", ".join(f"{GROUP_LABELS.get(g, g)} n={n.get(g, 0)}" for g in groups)


def main() -> None:
    apply_publication_style()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 7.95))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.30, 2.15, 1.55, 2.20],
                             left=0.085, right=0.945, top=0.970, bottom=0.048,
                             hspace=0.38)
    letters: list = []

    # ---- row 1: a | b (module scores for the two disease steps) -----------
    row1 = outer[0].subgridspec(1, 2, wspace=0.24)
    n_a = module_panel(fig, row1[0], ["IM", "MH_CD4"], T01, letters)
    n_b = module_panel(fig, row1[1], ["MH_CD4", "HLH"], T02, letters)
    NOTES.append(f"panels a,b: per-sample mean module scores; a {n_a}; b {n_b}; "
                 "two-sided Wilcoxon rank-sum, *** p<0.001, ** p<0.01, * p<0.05")
    NOTES.append("panels a,b: the upstream tables also contain Naive, Effector, "
                 "Exhausted and Proliferating columns that are empty for every "
                 "sample; those four modules are omitted rather than drawn blank")

    # ---- row 2: c | d (KEGG GSEA for the same two steps) ------------------
    # Leading columns are gutters: KEGG set names need more room than the default
    # figure margin provides.
    row2 = outer[1].subgridspec(1, 4, width_ratios=[0.80, 1.0, 0.80, 1.0],
                               wspace=0.09)
    gsea_c = pd.read_csv(T01 / "mh_cd4_vs_im_GSEA_top20.csv")
    ax_c = fig.add_subplot(row2[1])
    P.nes_barplot(ax_c, gsea_c, "Description", "NES", UP_COLOR, DOWN_COLOR,
                  max_label_chars=40)
    ax_c.set_title("MH-CD4 vs IM", fontsize=5.6, pad=2.5)
    letters.append(ax_c)

    gsea_d = pd.read_csv(T02 / "hlh_vs_mhcd4_GSEA_top20.csv")
    ax_d = fig.add_subplot(row2[3])
    P.nes_barplot(ax_d, gsea_d, "Description", "NES", UP_COLOR, DOWN_COLOR,
                  max_label_chars=40)
    ax_d.set_title("HLH vs MH-CD4", fontsize=5.6, pad=2.5)
    letters.append(ax_d)
    NOTES.append(f"panels c,d: top {len(gsea_c)} and {len(gsea_d)} KEGG sets by "
                 "adjusted p (padj<0.05), fgsea on the Wilcoxon-ranked gene list; "
                 "red = enriched in the more severe group")

    # ---- row 3: e | f | g -------------------------------------------------
    row3 = outer[2].subgridspec(1, 5, width_ratios=[0.58, 1.0, 0.56, 0.98, 1.05],
                               wspace=0.22)

    counts = pd.read_csv(T02 / "intersection_counts.csv").sort_values("n")
    set_label = {
        "MH_CD4_vs_IM_up": "MH-CD4 vs IM, up",
        "MH_CD4_vs_IM_down": "MH-CD4 vs IM, down",
        "HLH_vs_MH_CD4_up": "HLH vs MH-CD4, up",
        "HLH_up_and_MHCD4_down": "HLH up and MH-CD4 down",
        "intersection_sustained": "Sustained in both steps",
    }
    ax_e = fig.add_subplot(row3[1])
    P.count_barplot(ax_e, [set_label.get(s, s) for s in counts["set"]],
                    counts["n"].to_numpy(), SET_COLORS, xlabel="Genes")
    letters.append(ax_e)

    hp = pd.read_csv(T03 / "CD8_HP_ratio_per_sample.csv")
    ax_f = fig.add_subplot(row3[2])
    P.group_boxplots([ax_f], hp, ["HP_ratio"], "Newgroup6", GROUP_ORDER,
                     {"HP_ratio": "Homeostatic / progressive"},
                     ylabel="Ratio", title_fontsize=5.2)
    groups_present = [hp.loc[hp["Newgroup6"] == g, "HP_ratio"].dropna()
                      for g in GROUP_ORDER]
    p_hp = stats.kruskal(*[g for g in groups_present if len(g)]).pvalue
    ax_f.set_title(f"H/P ratio  {significance_stars(p_hp) or 'ns'}", fontsize=5.2,
                   pad=2.0)
    letters.append(ax_f)
    NOTES.append(f"panel f: {len(hp)} samples; Kruskal-Wallis across four groups "
                 f"p = {p_hp:.2g}")

    rev = pd.read_csv(T03 / "CD8_KEGG_reverse_pathways.csv")
    ax_g = fig.add_subplot(row3[4])
    P.paired_nes_bars(ax_g, rev, "term", ["NES_homeostatic", "NES_progressive"],
                      [UP_COLOR, DOWN_COLOR],
                      ["Homeostatic axis", "Progressive axis"], max_label_chars=40)
    ax_g.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.0,
                handlelength=1.2, handletextpad=0.4, borderpad=0.2,
                columnspacing=1.0)
    letters.append(ax_g)
    NOTES.append(f"panel g: {len(rev)} KEGG sets whose enrichment sign flips "
                 "between the homeostatic and progressive contrasts "
                 f"(padj cutoff {rev['padj_cutoff_used'].iloc[0]})")

    # ---- row 4: h | i -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 3, width_ratios=[1.0, 1.15, 0.030], wspace=0.42)

    trend = pd.read_csv(T05 / "CD8_pathway_trend.csv")
    top = pd.concat([
        trend[trend["class"] == "monotonic_increasing"].head(10),
        trend[trend["class"] == "monotonic_decreasing"].head(10),
    ])
    ax_h = fig.add_subplot(row4[0])
    P.trend_lines(ax_h, top, "pathway", ["mean_IM", "mean_MH_CD4", "mean_HLH"],
                  [GROUP_LABELS[g] for g in ("IM", "MH_CD4", "HLH")],
                  class_col="class",
                  class_colors={"monotonic_increasing": UP_COLOR,
                                "monotonic_decreasing": DOWN_COLOR},
                  n_label=2)
    ax_h.set_xlim(-0.12, 3.55)
    handles = [Line2D([], [], color=UP_COLOR, linewidth=0.8, label="Increasing"),
               Line2D([], [], color=DOWN_COLOR, linewidth=0.8, label="Decreasing")]
    ax_h.legend(handles=handles, loc="upper left", fontsize=5.0, handlelength=1.2,
                handletextpad=0.4, borderpad=0.2)
    letters.append(ax_h)
    n_inc = int((trend["class"] == "monotonic_increasing").sum())
    n_dec = int((trend["class"] == "monotonic_decreasing").sum())
    NOTES.append(f"panel h: 10 of {n_inc} increasing and 10 of {n_dec} decreasing "
                 "KEGG sets (|Spearman rho| > 0.9 across the three disease groups); "
                 "the three largest movers per direction are labelled")

    cells = pd.read_csv(DATA / "fig3_cells.csv")
    genes = [g for g in INC_GENES + DEC_GENES if g in cells.columns]
    means = np.array([[cells.loc[cells["group"] == grp, g].mean()
                       for grp in GROUP_ORDER] for g in genes])
    z = np.nan_to_num((means - means.mean(axis=1, keepdims=True))
                      / (means.std(axis=1, keepdims=True) + 1e-9))
    inc_rows = [i for i, g in enumerate(genes) if g in INC_GENES]
    dec_rows = [i for i, g in enumerate(genes) if g in DEC_GENES]
    cax_i = fig.add_subplot(row4[2])
    ax_i = P.zscore_heatmap(fig, row4[1], cax_i, z, genes,
                            [GROUP_LABELS[g] for g in GROUP_ORDER],
                            [inc_rows, dec_rows],
                            block_titles=["Increasing", "Decreasing"])
    letters.append(ax_i)
    NOTES.append(f"panel i: top 20 increasing and top 20 decreasing genes by "
                 f"per-cell Spearman rho; group-mean expression over "
                 f"{len(cells):,} CD8+ cells, z-scored across the four groups")

    for ax, lab in zip(letters, "abcdefghi"):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure3")
    (OUT / "Figure3_provenance.txt").write_text(
        "Figure 3 | Opposing recovery and exhaustion programmes in CD8+ T cells\n"
        "Rendered in Python (matplotlib) from the noIMM regeneration tables;\n"
        "R used only to export per-cell expression for panel i.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
