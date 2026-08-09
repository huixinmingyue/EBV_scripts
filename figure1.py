"""Figure 1 | Single-cell immune landscape across the paediatric EBV continuum.

Core conclusion: the peripheral immune compartment is progressively remodelled
along the clinical continuum HC -> IM -> MH-CD4 -> HLH, and the lineage shifts
track clinical severity.

Panels
  a  UMAP of 338,691 cells annotated into 13 immune lineages
  b  Canonical lineage markers projected on the same embedding
  c  Marker specificity per lineage (mean scaled expression, % expressing)
  d  Per-patient lineage proportions across the four groups
  e  Clinical severity metrics per patient, ordered by composite severity
  f  Within-group correlation between lineage proportion and severity
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

from nature_style import (
    CELLTYPE_COLORS,
    DIVERGING_CMAP,
    EXPR_CMAP,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    HEIGHT_MAX,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    WIDTH_DOUBLE,
    apply_publication_style,
    corner_arrows,
    embedding_axes,
    halo_text,
    panel_label,
    save_figure,
    significance_stars,
    strip_axes,
)

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data" / "fig1"
PROP = Path("F:/EBV/EBV_xiaomi/EBV/EBV_document/proportion")
OUT = ROOT / "Figure1"

GENES = ["PTPRC", "CD3D", "CD4", "CD8A", "NCAM1", "FCGR3A", "CD19", "CD14"]

# Compact lineage names keep dense panels legible at journal width; the full
# names stay in the figure legend.
SHORT = {
    "Neutrophil/Eosinophil": "Neut/Eos",
    "Erythroid_cells": "Erythroid",
    "Plasma cells": "Plasma",
    "gdT_cells": "gdT",
}
NOTES: list[str] = []


def load_cells() -> pd.DataFrame:
    df = pd.read_csv(DATA / "fig1_cells.csv")
    NOTES.append(f"panel a-c: {len(df):,} cells, {df['cell_type3'].nunique()} lineages, "
                 f"groups {sorted(df['Newgroup6'].unique())}")
    return df


def _repel(points: np.ndarray, min_dist: float, iterations: int = 260) -> np.ndarray:
    """Push overlapping label anchors apart while keeping them near the origin."""
    pos = points.astype(float).copy()
    for _ in range(iterations):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                delta = pos[i] - pos[j]
                dist = float(np.hypot(*delta))
                if dist < 1e-9:
                    delta = np.array([1e-3, 1e-3])
                    dist = float(np.hypot(*delta))
                if dist < min_dist:
                    push = (min_dist - dist) / 2.0
                    step = delta / dist * push
                    pos[i] += step
                    pos[j] -= step
                    moved = True
        if not moved:
            break
    return pos


def panel_a(fig, ax, cells: pd.DataFrame) -> None:
    order = cells["cell_type3"].value_counts().index.tolist()
    for ct in order:
        sub = cells[cells["cell_type3"] == ct]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.6, linewidths=0,
                   color=CELLTYPE_COLORS.get(ct, NEUTRAL_MID), alpha=0.75,
                   rasterized=True)
    embedding_axes(ax)

    anchors = np.array([[cells.loc[cells["cell_type3"] == ct, "UMAP1"].median(),
                         cells.loc[cells["cell_type3"] == ct, "UMAP2"].median()]
                        for ct in order])
    span = max(np.ptp(anchors[:, 0]), np.ptp(anchors[:, 1]))
    placed = _repel(anchors, min_dist=span * 0.22)
    for ct, anchor, label_pos in zip(order, anchors, placed):
        text = SHORT.get(ct, ct.replace("_", " "))
        if np.hypot(*(label_pos - anchor)) > span * 0.02:
            ax.plot([anchor[0], label_pos[0]], [anchor[1], label_pos[1]],
                    color=NEUTRAL_MID, linewidth=0.25, zorder=4)
        halo_text(ax, label_pos[0], label_pos[1], text, fontsize=5.0)
    ax.margins(0.16)
    corner_arrows(ax)
    ax.set_title(f"{len(cells):,} cells  |  13 lineages", fontsize=5.2,
                 color=NEUTRAL_MID, pad=1.5)


def panel_b(fig, axes, cells: pd.DataFrame) -> None:
    for ax, gene in zip(axes, GENES):
        vals = cells[gene].to_numpy()
        hi = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
        hi = hi if hi > 0 else 1.0
        order = np.argsort(vals)  # draw expressing cells last so they stay visible
        ax.scatter(cells["UMAP1"].to_numpy()[order], cells["UMAP2"].to_numpy()[order],
                   c=np.clip(vals[order], 0, hi), cmap=EXPR_CMAP, s=0.35, linewidths=0,
                   vmin=0, vmax=hi, rasterized=True)
        embedding_axes(ax)
        ax.set_title(gene, fontsize=5.4, style="italic", pad=1.0)
    NOTES.append("panel b: per-gene colour scale 0 to 99th percentile of non-zero "
                 "expression; all cells plotted, none downsampled")


def panel_c(fig, ax, cax, lax, cells: pd.DataFrame) -> None:
    order = cells["cell_type3"].value_counts().index.tolist()
    mean_expr = np.zeros((len(GENES), len(order)))
    pct_expr = np.zeros_like(mean_expr)
    for j, ct in enumerate(order):
        sub = cells[cells["cell_type3"] == ct]
        for i, g in enumerate(GENES):
            v = sub[g].to_numpy()
            mean_expr[i, j] = v.mean()
            pct_expr[i, j] = (v > 0).mean() * 100
    # z-score each marker across lineages so colour encodes relative specificity
    z = (mean_expr - mean_expr.mean(axis=1, keepdims=True)) / (
        mean_expr.std(axis=1, keepdims=True) + 1e-9)
    vmax = float(np.abs(z).max())

    size_scale = 0.42
    xs, ys = np.meshgrid(np.arange(len(order)), np.arange(len(GENES)))
    sc = ax.scatter(xs.ravel(), ys.ravel(), s=pct_expr.ravel() * size_scale,
                    c=z.ravel(), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                    linewidths=0.15, edgecolors="white")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([SHORT.get(c, c.replace("_", " ")) for c in order], rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(GENES)))
    ax.set_yticklabels(GENES, style="italic")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(-0.7, len(GENES) - 0.3)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)

    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("z-score", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)

    strip_axes(lax)
    handles = [Line2D([], [], marker="o", linestyle="none",
                      markersize=np.sqrt(p * size_scale), markerfacecolor=NEUTRAL_MID,
                      markeredgecolor="none", label=f"{p:g}")
               for p in (25, 50, 75, 100)]
    leg = lax.legend(handles=handles, title="% cells expressing", loc="center left",
                     handletextpad=0.5, labelspacing=0.7, borderpad=0.2, fontsize=5.0)
    leg.get_title().set_fontsize(5.0)


def panel_d(fig, axes, prop: pd.DataFrame) -> None:
    order = (prop.groupby("cell_type3")["count"].sum()
             .sort_values(ascending=False).index.tolist())
    for ax, ct in zip(axes, order):
        sub = prop[prop["cell_type3"] == ct]
        data, colors = [], []
        for g in GROUP_ORDER:
            data.append(sub[sub["Newgroup6"] == g]["proportion"].to_numpy())
            colors.append(GROUP_COLORS[g])
        bp = ax.boxplot(data, widths=0.62, showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=0.7),
                        boxprops=dict(linewidth=0.3, edgecolor=NEUTRAL_DARK),
                        whiskerprops=dict(linewidth=0.3, color=NEUTRAL_DARK),
                        capprops=dict(linewidth=0.3, color=NEUTRAL_DARK))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        rng = np.random.default_rng(42)
        for i, (vals, c) in enumerate(zip(data, colors), start=1):
            ax.scatter(i + rng.uniform(-0.14, 0.14, size=len(vals)), vals, s=1.4,
                       color=NEUTRAL_DARK, alpha=0.7, linewidths=0, zorder=3)
        groups = [d for d in data if len(d) > 0]
        star_txt = ""
        if len(groups) >= 2:
            p = stats.kruskal(*groups).pvalue
            star = significance_stars(p)
            star_txt = star if star else "ns"
        title = SHORT.get(ct, ct.replace("_", " "))
        ax.set_title(f"{title}  {star_txt}".strip(), fontsize=5.0, pad=2.0)
        ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.tick_params(length=1.2, pad=1.0)
        ax.margins(y=0.14)
    for idx, ax in enumerate(axes[:len(order)]):
        if idx % 5 == 0:
            ax.set_ylabel("Proportion (%)", fontsize=5.0, labelpad=1.5)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    NOTES.append("panel d: Kruskal-Wallis across the four groups per lineage; "
                 "*** p<0.001, ** p<0.01, * p<0.05, ns not significant")


def panel_e(fig, ax, cax) -> None:
    df = pd.read_csv(PROP / "severity_clinical_4groups_by_severity.csv")
    before = len(df)
    df = df[df["Newgroup6"].isin(GROUP_ORDER)].copy()
    NOTES.append(f"panel e: {before} patients in source, {len(df)} retained after "
                 f"restricting to {GROUP_ORDER} (IM_M and MH_CD56 excluded to match "
                 "the four-group continuum used in panels a-d)")
    df = df.sort_values("severity_score_4")
    metrics = [("log10_ebv", "EBV DNA"), ("log10_ferritin", "Ferritin"),
               ("log10_scd25", "sCD25"), ("lymph_num", "Lymphocytes")]
    mat = np.full((len(metrics), len(df)), np.nan)
    for i, (col, _) in enumerate(metrics):
        v = df[col].to_numpy(dtype=float)
        m, s = np.nanmean(v), np.nanstd(v)
        mat[i] = (v - m) / (s if s > 0 else 1.0)
    vmax = float(np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([lab for _, lab in metrics])
    ax.set_xticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # group ribbon beneath the heatmap marks the continuum ordering
    y = len(metrics) - 0.42
    ribbon_h = 0.34
    for x, g in enumerate(df["Newgroup6"]):
        ax.add_patch(plt.Rectangle((x - 0.5, y), 1, ribbon_h, color=GROUP_COLORS[g],
                                   clip_on=False, linewidth=0))
    ax.set_ylim(y + ribbon_h + 0.06, -0.5)
    ax.text(-0.9, y + ribbon_h / 2, "Group", ha="right", va="center", fontsize=5.0)
    ax.set_xlabel(f"Patients ordered by composite severity score (n = {len(df)})",
                  fontsize=5.0, labelpad=3)
    n_missing = int(np.isnan(mat).sum())
    if n_missing:
        NOTES.append(f"panel e: {n_missing} of {mat.size} metric values missing in "
                     "source and left blank rather than imputed")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z-score", fontsize=5.0, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)


def panel_f(fig, ax, cax) -> None:
    df = pd.read_csv(PROP / "celltype3_severity_correlation_by_group_4continuum.csv")
    before = sorted(df["Newgroup6"].unique())
    groups = [g for g in ["IM", "MH_CD4", "HLH"] if g in before]
    df = df[df["Newgroup6"].isin(groups)]
    NOTES.append(f"panel f: source groups {before}; retained {groups} (IM_M excluded "
                 "for consistency; HC carries no severity score)")
    cts = (df.groupby("cell_type3")["correlation"].mean()
           .reindex(sorted(df["cell_type3"].unique())).index.tolist())
    mat = np.full((len(cts), len(groups)), np.nan)
    pmat = np.ones_like(mat)
    for i, ct in enumerate(cts):
        for j, g in enumerate(groups):
            row = df[(df["cell_type3"] == ct) & (df["Newgroup6"] == g)]
            if len(row):
                mat[i, j] = row["correlation"].iloc[0]
                pmat[i, j] = row["p_value"].iloc[0]
    im = ax.imshow(mat, cmap=DIVERGING_CMAP, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups])
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels([SHORT.get(c, c.replace("_", " ")) for c in cts])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(cts)):
        for j in range(len(groups)):
            star = significance_stars(pmat[i, j])
            if star:
                ax.text(j, i, star, ha="center", va="center", fontsize=5.0, color="white")
    n_per_group = {g: int(df[df["Newgroup6"] == g]["n"].iloc[0]) for g in groups}
    ax.set_xlabel("  ".join(f"{GROUP_LABELS[g]} n={n}" for g, n in n_per_group.items()),
                  fontsize=5.0, labelpad=2)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman r\n(proportion vs severity)", fontsize=5.0, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)


def main() -> None:
    apply_publication_style()
    cells = load_cells()
    prop = pd.read_csv(DATA / "fig1_patient_proportions.csv")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 9.25))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.95, 1.60, 2.85, 2.05],
                             left=0.075, right=0.955, top=0.975, bottom=0.040,
                             hspace=0.34)

    # ---- row 1: a | b -----------------------------------------------------
    row1 = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.72], wspace=0.06)
    ax_a = fig.add_subplot(row1[0])
    panel_a(fig, ax_a, cells)
    grid_b = row1[1].subgridspec(2, 4, wspace=0.06, hspace=0.16)
    axes_b = [fig.add_subplot(grid_b[i, j]) for i in range(2) for j in range(4)]
    panel_b(fig, axes_b, cells)

    # ---- row 2: c + right sidebar (colour scale above size key) -----------
    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 0.014, 0.145], wspace=0.035)
    ax_c = fig.add_subplot(row2[0])
    side = row2[2].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.25)
    cax_c = fig.add_subplot(row2[1])
    lax_c = fig.add_subplot(side[1])
    panel_c(fig, ax_c, cax_c, lax_c, cells)

    # ---- row 3: d ---------------------------------------------------------
    grid_d = outer[2].subgridspec(3, 5, wspace=0.46, hspace=1.05)
    axes_d = [fig.add_subplot(grid_d[i, j]) for i in range(3) for j in range(5)]
    panel_d(fig, axes_d, prop)

    # ---- row 4: e | f -----------------------------------------------------
    row4 = outer[3].subgridspec(1, 5, width_ratios=[1.0, 0.02, 0.30, 0.42, 0.02],
                               wspace=0.10)
    ax_e = fig.add_subplot(row4[0])
    cax_e = fig.add_subplot(row4[1])
    ax_f = fig.add_subplot(row4[3])
    cax_f = fig.add_subplot(row4[4])
    panel_e(fig, ax_e, cax_e)
    panel_f(fig, ax_f, cax_f)

    # shared group legend for panels d and e
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=3,
                      markerfacecolor=GROUP_COLORS[g], markeredgecolor="none",
                      label=GROUP_LABELS[g]) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.0),
               ncol=4, fontsize=5.0, handletextpad=0.4, columnspacing=1.2)

    for ax, lab in ((ax_a, "a"), (axes_b[0], "b"), (ax_c, "c"), (axes_d[0], "d"),
                    (ax_e, "e"), (ax_f, "f")):
        panel_label(fig, ax, lab)

    written = save_figure(fig, OUT, "Figure1")
    (OUT / "Figure1_provenance.txt").write_text(
        "Figure 1 | Single-cell immune landscape across the paediatric EBV continuum\n"
        "Rendered in Python (matplotlib); R used only to export data from the Seurat object.\n\n"
        + "\n".join(f"- {n}" for n in NOTES) + "\n",
        encoding="utf-8")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
