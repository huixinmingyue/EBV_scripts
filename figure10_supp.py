"""Figure 10 supplement 1 | The gene-subset reading of the B-cell signalling
trend, shown alongside the diagnostic that motivates using the pathway-level
reading in Figure 10e,f.

Two ways of asking "which KEGG signalling pathways track the IM to MH-CD4 to
HLH continuum" were available in the archive and they do not measure the same
thing:

  * Figure 10e,f take the pathway as the unit. All genes annotated to a pathway
    are averaged and the pathway is kept only if that single mean is strictly
    monotonic across the three groups. Four pathways rise and 24 fall.
  * This supplement takes the gene subset as the unit. Within every pathway the
    genes are first split into those that rise and those that fall, and the two
    subsets are averaged separately. Any pathway holding at least a couple of
    genes of each kind therefore appears in both directions.

The second reading is what the archived summary tables encode, and panels a and
b reproduce it. Panel c is the reason it is not used for the main figure: all 24
pathways in the rising table also sit in the falling table, the enrichment
statistic is nearly identical between the two, and for most pathways the falling
gene subset is several times larger than the rising one.

Panels
  a  Pathways ranked by NES with only their rising genes averaged
  b  The same pathways with only their falling genes averaged
  c  Genes contributing to each direction, and the shared NES
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import panels as P
from nature_style import (
    GROUP_COLORS,
    GROUP_LABELS,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    panel_label,
    save_figure,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
RDATA = Path("F:/EBV/EBV_xiaomi/EBV/Rdata")
FIG10 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig10")
OUT = ROOT / "Figure10"

TREND_GROUPS = ["IM", "MH_CD4", "HLH"]
N_SHOW = 8
UP_COLOR = "#C0392B"
DOWN_COLOR = "#4E79A7"

NOTES: list[str] = []


def load(direction: str) -> pd.DataFrame:
    df = pd.read_csv(
        RDATA / f"bcells_new_im_mhcd4_hlh_{direction}_signaling_pathway_trend_summary.csv")
    # Every entry is a signalling pathway, so carrying the words on each tick
    # only costs the space the names need.
    df["pathway"] = df["pathway"].str.replace(r"\s+signaling pathway$", "",
                                              regex=True)
    return df


def main() -> None:
    apply_publication_style()

    inc = load("increasing")
    dec = load("decreasing")
    pathway_level = pd.read_csv(FIG10 / "TC_EBV_05_B_monotonic" / "B_pathway_trend.csv")

    order = inc.sort_values("NES", ascending=False)["pathway"].head(N_SHOW).tolist()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 2.45))
    # The leading column is an empty gutter so the pathway names in a can sit
    # outside the plot area without colliding with the panel letter.
    grid = fig.add_gridspec(1, 4, width_ratios=[0.34, 1.00, 0.78, 0.94],
                            left=0.008, right=0.988, top=0.800, bottom=0.150,
                            wspace=0.15)

    axes = {}
    for idx, (df, key, title) in enumerate(
            [(inc, "a", "Rising genes only"),
             (dec, "b", "Falling genes only")]):
        ax = fig.add_subplot(grid[idx + 1])
        axes[key] = ax
        long = df.melt(id_vars=["pathway"], value_vars=TREND_GROUPS,
                       var_name="group", value_name="mean_expr")
        P.grouped_hbars(ax, long, "pathway", "group", "mean_expr", order,
                        TREND_GROUPS,
                        {g: GROUP_COLORS[g] for g in TREND_GROUPS},
                        xlabel="Mean expression", max_label_chars=24)
        ax.set_title(title, fontsize=5.2, pad=2.0, loc="left")
        ax.tick_params(axis="y", length=0)
        if key == "b":
            ax.set_yticklabels([])
            ax.legend(handles=[Patch(facecolor=GROUP_COLORS[g], edgecolor="none",
                                     label=GROUP_LABELS[g]) for g in TREND_GROUPS],
                      loc="lower right", fontsize=5.0, handlelength=1.0,
                      handletextpad=0.35, labelspacing=0.25, borderpad=0.25,
                      frameon=False)

    # ---- c: what went into each direction ---------------------------------
    counts = (inc[["pathway", "n_genes", "NES"]]
              .merge(dec[["pathway", "n_genes"]], on="pathway",
                     suffixes=("_up", "_down"))
              .set_index("pathway").reindex(order))
    ax_c = fig.add_subplot(grid[3])
    axes["c"] = ax_c
    y = np.arange(len(order))
    ax_c.barh(y, counts["n_genes_up"], height=0.62, color=UP_COLOR, linewidth=0)
    ax_c.barh(y, -counts["n_genes_down"], height=0.62, color=DOWN_COLOR,
              linewidth=0)
    ax_c.axvline(0, color=NEUTRAL_DARK, linewidth=0.5)
    for yi, (up, down, nes) in enumerate(zip(counts["n_genes_up"],
                                             counts["n_genes_down"],
                                             counts["NES"])):
        ax_c.text(up + 1.0, yi, f"{int(up)}", va="center", ha="left",
                  fontsize=5.0, color=UP_COLOR)
        ax_c.text(-down - 1.0, yi, f"{int(down)}", va="center", ha="right",
                  fontsize=5.0, color=DOWN_COLOR)
        ax_c.text(0.985, yi, f"NES {nes:.2f}", transform=ax_c.get_yaxis_transform(),
                  va="center", ha="right", fontsize=5.0, color=NEUTRAL_MID)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.set_ylim(len(order) - 0.45, -0.55)
    lim = float(counts["n_genes_down"].max()) * 1.18
    ax_c.set_xlim(-lim, lim * 0.92)
    ax_c.set_xticks([-30, -20, -10, 0, 10])
    ax_c.set_xticklabels(["30", "20", "10", "0", "10"])
    ax_c.set_xlabel("Genes averaged into each direction", labelpad=1.5)
    ax_c.tick_params(length=1.2, pad=1.0)
    ax_c.tick_params(axis="y", length=0)
    ax_c.spines["left"].set_visible(False)
    ax_c.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax_c.set_axisbelow(True)
    ax_c.text(0.0, 1.02, "Falling", transform=ax_c.transAxes, fontsize=5.2,
              color=DOWN_COLOR, ha="left", va="bottom")
    ax_c.text(0.58, 1.02, "Rising", transform=ax_c.transAxes, fontsize=5.2,
              color=UP_COLOR, ha="left", va="bottom")

    n_both = len(set(inc["pathway"]) & set(dec["pathway"]))
    nes_gap = float((counts["NES"]
                     - dec.set_index("pathway").reindex(order)["NES"]).abs().max())
    n_up_level = int((pathway_level["class"] == "monotonic_increasing").sum())
    n_down_level = int((pathway_level["class"] == "monotonic_decreasing").sum())

    NOTES.append(f"panels a,b: the {N_SHOW} pathways with the highest NES in the "
                 "archived rising table, drawn from bcells_new_im_mhcd4_hlh_"
                 "{increasing,decreasing}_signaling_pathway_trend_summary.csv; "
                 "bars are the mean expression of the direction-selected gene "
                 "subset in each group")
    NOTES.append(f"panel c: all {n_both} of the {len(inc)} pathways in the rising "
                 f"table also appear among the {len(dec)} in the falling table, and "
                 "the NES differs between the two by at most "
                 f"{nes_gap:.4f}, so the statistic describes the pathway rather "
                 "than the direction")
    NOTES.append("every pathway in both tables carries Adjusted.P.value = 1, so "
                 "none of these trends is significant after multiple-testing "
                 "correction")
    NOTES.append("Figure 10e,f instead require the mean over all genes in a "
                 f"pathway to be monotonic, which admits {n_up_level} rising and "
                 f"{n_down_level} falling pathways; the two readings are not "
                 "comparable and the counts should not be quoted against each other")

    # Panel a's letter sits over its own label gutter; the clamp in panel_label
    # keeps it inside the canvas.
    for key, dx in (("a", -0.20), ("b", -0.030), ("c", -0.030)):
        panel_label(fig, axes[key], key, dx=dx, dy=0.058)

    fig.text(0.008, 0.978,
             "Figure 10 supplement 1 | Gene-subset reading of the B-cell "
             "signalling trend, and why the main figure uses the pathway-level one",
             fontsize=6.0, fontweight="bold", ha="left", va="top")

    written = save_figure(fig, OUT, "Figure10_supp1")
    (OUT / "Figure10_supp1_provenance.txt").write_text(
        "Figure 10 supplement 1 | Gene-subset reading of the B-cell signalling "
        "trend\nRendered in Python (matplotlib) from archived summary tables; no "
        "recomputation from the Seurat object.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n", encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
