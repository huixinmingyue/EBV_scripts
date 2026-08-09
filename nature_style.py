"""Shared publication style and panel helpers for the EBV manuscript figures.

Rendering is Python-only (matplotlib). R is used upstream for data export only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1.0 / 25.4
WIDTH_DOUBLE = 180 * MM
WIDTH_SINGLE = 89 * MM
HEIGHT_MAX = 240 * MM

# Disease continuum, ordered from healthy to most severe.
GROUP_ORDER = ["HC", "IM", "MH_CD4", "HLH"]
GROUP_LABELS = {"HC": "HC", "IM": "IM", "MH_CD4": "MH-CD4", "HLH": "HLH"}
GROUP_COLORS = {
    "HC": "#4C7FB8",
    "IM": "#7BAA5B",
    "MH_CD4": "#E28E2C",
    "HLH": "#C0392B",
}

# Lineage colours follow the manuscript's established cell-type code so that
# every figure in the series stays mutually recognisable.
CELLTYPE_COLORS = {
    "CD8T": "#4DAF4A",
    "CD4T": "#377EB8",
    "Neutrophil/Eosinophil": "#A65628",
    "Mono/Mac": "#D9B310",
    "Bcells": "#E41A1C",
    "gdT_cells": "#984EA3",
    "NK": "#F781BF",
    "NKT": "#FF7F00",
    "Erythroid_cells": "#8DA0CB",
    "cDC": "#999999",
    "Platelets": "#E78AC3",
    "Plasma cells": "#FC8D62",
    "pDC": "#66C2A5",
}

NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#767676"

# Sequential ramp for expression overlays; light grey floor keeps empty cells
# visually inert without hiding them.
EXPR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "expr", ["#DEDEDE", "#BCBDDC", "#8073AC", "#54278F"]
)
# Diverging ramp for z-scores and correlations, balanced about zero.
DIVERGING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "diverging", ["#2C6FAD", "#8EB8D8", "#F2F2F2", "#E2A17A", "#B02418"]
)


def apply_publication_style(font_size: float = 6.0, axes_linewidth: float = 0.5) -> None:
    """Nature-style rcParams for dense journal-width multi-panel figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size - 0.5
    plt.rcParams["ytick.labelsize"] = font_size - 0.5
    plt.rcParams["legend.fontsize"] = font_size - 0.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["xtick.major.size"] = 1.6
    plt.rcParams["ytick.major.size"] = 1.6
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def panel_label(fig, ax, label: str, dx: float = -0.055, dy: float = 0.012) -> None:
    """Place a bold panel letter just outside the top-left corner of an axes."""
    box = ax.get_position()
    fig.text(
        max(box.x0 + dx, 0.002),
        min(box.y1 + dy, 0.998),
        label,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def strip_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def embedding_axes(ax) -> None:
    """Minimal corner-arrow frame used for UMAP panels."""
    strip_axes(ax)
    ax.set_aspect("equal")


def corner_arrows(ax, xlabel: str = "UMAP1", ylabel: str = "UMAP2", frac: float = 0.18) -> None:
    """Draw short axis arrows in the lower-left corner instead of full spines."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lx = (x1 - x0) * frac
    ly = (y1 - y0) * frac
    ox = x0 + (x1 - x0) * 0.02
    oy = y0 + (y1 - y0) * 0.02
    arrow = dict(arrowstyle="-|>", color=NEUTRAL_DARK, linewidth=0.5, mutation_scale=4)
    ax.annotate("", xy=(ox + lx, oy), xytext=(ox, oy), arrowprops=arrow)
    ax.annotate("", xy=(ox, oy + ly), xytext=(ox, oy), arrowprops=arrow)
    ax.text(ox + lx / 2, oy - (y1 - y0) * 0.035, xlabel, ha="center", va="top", fontsize=5.0,
            color=NEUTRAL_DARK)
    ax.text(ox - (x1 - x0) * 0.035, oy + ly / 2, ylabel, ha="right", va="center", rotation=90,
            fontsize=5.0, color=NEUTRAL_DARK)


def halo_text(ax, x, y, s, fontsize=5.0, color="black", weight="bold", **kw):
    """Text with a white outline so labels stay readable over dense scatters."""
    from matplotlib import patheffects

    t = ax.text(x, y, s, fontsize=fontsize, color=color, fontweight=weight,
                ha="center", va="center", **kw)
    t.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="white")])
    return t


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def add_sig_bracket(ax, x1, x2, y, text, lw=0.4, tick=None, fontsize=5.0):
    """Draw a significance bracket spanning x1..x2 at height y."""
    if tick is None:
        span = ax.get_ylim()
        tick = (span[1] - span[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], lw=lw, color=NEUTRAL_DARK,
            solid_joinstyle="miter", clip_on=False)
    ax.text((x1 + x2) / 2, y + tick, text, ha="center", va="bottom", fontsize=fontsize,
            color=NEUTRAL_DARK, clip_on=False)


def save_figure(fig, out_dir: Path, stem: str, dpi: int = 600, qa_pdf: bool = True) -> list[str]:
    """Export SVG (editable vector) and PNG at review resolution.

    A PDF copy goes to a `_qa` subfolder so the rendered glyph sizes can be
    audited against the 5 pt floor; it is not part of the delivered set.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path)
    written.append(str(svg_path))
    fig.savefig(png_path, dpi=dpi)
    written.append(str(png_path))
    if qa_pdf:
        qa_dir = out_dir / "_qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(qa_dir / f"{stem}.pdf")
    plt.close(fig)
    return written


def dot_legend_handles(sizes, labels, color=NEUTRAL_MID):
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(s),
               markerfacecolor=color, markeredgecolor="none", label=lab)
        for s, lab in zip(sizes, labels)
    ]
