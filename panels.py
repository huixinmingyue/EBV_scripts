"""Reusable panel builders shared by the EBV manuscript figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from nature_style import (
    DIVERGING_CMAP,
    EXPR_CMAP,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    corner_arrows,
    embedding_axes,
    halo_text,
    significance_stars,
    strip_axes,
)


def repel(points: np.ndarray, min_dist: float, iterations: int = 260) -> np.ndarray:
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
                    step = delta / dist * ((min_dist - dist) / 2.0)
                    pos[i] += step
                    pos[j] -= step
                    moved = True
        if not moved:
            break
    return pos


def umap_by_category(ax, cells, category_col, colors, label_map=None, size=0.6,
                     label=True, min_dist_frac=0.20, label_fontsize=5.0):
    """Scatter an embedding coloured by a categorical annotation, with direct labels."""
    order = cells[category_col].value_counts().index.tolist()
    for cat in order:
        sub = cells[cells[category_col] == cat]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=size, linewidths=0,
                   color=colors.get(cat, NEUTRAL_MID), alpha=0.75, rasterized=True)
    embedding_axes(ax)
    if label:
        anchors = np.array([[cells.loc[cells[category_col] == c, "UMAP1"].median(),
                             cells.loc[cells[category_col] == c, "UMAP2"].median()]
                            for c in order])
        span = max(np.ptp(anchors[:, 0]), np.ptp(anchors[:, 1]))
        placed = repel(anchors, min_dist=span * min_dist_frac)
        for cat, anchor, pos in zip(order, anchors, placed):
            text = label_map.get(cat, cat) if label_map else cat
            if np.hypot(*(pos - anchor)) > span * 0.02:
                ax.plot([anchor[0], pos[0]], [anchor[1], pos[1]], color=NEUTRAL_MID,
                        linewidth=0.25, zorder=4)
            halo_text(ax, pos[0], pos[1], text, fontsize=label_fontsize)
    ax.margins(0.16)
    corner_arrows(ax)
    return order


def umap_expression(ax, cells, gene, size=0.35, hi_pct=99):
    """Scatter an embedding coloured by expression of one gene."""
    vals = cells[gene].to_numpy()
    hi = np.percentile(vals[vals > 0], hi_pct) if (vals > 0).any() else 1.0
    hi = hi if hi > 0 else 1.0
    order = np.argsort(vals)  # expressing cells drawn last so they stay visible
    ax.scatter(cells["UMAP1"].to_numpy()[order], cells["UMAP2"].to_numpy()[order],
               c=np.clip(vals[order], 0, hi), cmap=EXPR_CMAP, s=size, linewidths=0,
               vmin=0, vmax=hi, rasterized=True)
    embedding_axes(ax)
    ax.set_title(gene, fontsize=5.4, style="italic", pad=1.0)


def dotplot(fig, ax, cax, lax, cells, genes, category_col, category_order,
            label_map=None, size_scale=0.42, genes_on_x=True):
    """Mean scaled expression (colour) and per-cent expressing (size)."""
    mean_expr = np.zeros((len(genes), len(category_order)))
    pct_expr = np.zeros_like(mean_expr)
    for j, cat in enumerate(category_order):
        sub = cells[cells[category_col] == cat]
        for i, g in enumerate(genes):
            v = sub[g].to_numpy()
            mean_expr[i, j] = v.mean() if len(v) else 0.0
            pct_expr[i, j] = (v > 0).mean() * 100 if len(v) else 0.0
    z = (mean_expr - mean_expr.mean(axis=1, keepdims=True)) / (
        mean_expr.std(axis=1, keepdims=True) + 1e-9)
    vmax = float(np.abs(z).max())

    if genes_on_x:
        z, pct_expr = z.T, pct_expr.T
        rows, cols = category_order, genes
        row_labels = [label_map.get(c, c) if label_map else c for c in category_order]
        col_labels = genes
        italic_cols, italic_rows = True, False
    else:
        rows, cols = genes, category_order
        row_labels = genes
        col_labels = [label_map.get(c, c) if label_map else c for c in category_order]
        italic_cols, italic_rows = False, True

    xs, ys = np.meshgrid(np.arange(len(cols)), np.arange(len(rows)))
    sc = ax.scatter(xs.ravel(), ys.ravel(), s=pct_expr.ravel() * size_scale,
                    c=z.ravel(), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                    linewidths=0.15, edgecolors="white")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", rotation_mode="anchor",
                       style="italic" if italic_cols else "normal")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels, style="italic" if italic_rows else "normal")
    ax.set_xlim(-0.7, len(cols) - 0.3)
    ax.set_ylim(-0.7, len(rows) - 0.3)
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


def stacked_violin(axes, cells, genes, category_col, category_order, colors,
                   label_map=None):
    """Seurat-style stacked violin: one gene per row, one category per column."""
    for row, (ax, gene) in enumerate(zip(axes, genes)):
        data, face = [], []
        for cat in category_order:
            data.append(cells.loc[cells[category_col] == cat, gene].to_numpy())
            face.append(colors.get(cat, NEUTRAL_MID))
        parts = ax.violinplot(data, positions=np.arange(len(category_order)),
                              widths=0.85, showextrema=False, showmedians=False)
        for body, c in zip(parts["bodies"], face):
            body.set_facecolor(c)
            body.set_alpha(0.9)
            body.set_linewidth(0)
        ax.set_yticks([])
        ax.set_ylabel(gene, fontsize=5.0, style="italic", rotation=0, ha="right",
                      va="center", labelpad=2)
        ax.set_xlim(-0.65, len(category_order) - 0.35)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)
        if row == len(genes) - 1:
            ax.set_xticks(np.arange(len(category_order)))
            ax.set_xticklabels(
                [label_map.get(c, c) if label_map else c for c in category_order],
                rotation=40, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)


def proportion_boxplots(axes, prop, category_order, label_map=None,
                        group_order=None, value_col="proportion",
                        category_col="celltype", ylabel="Proportion (%)", ncol=5):
    """Per-patient proportion boxplots, one panel per cell category."""
    group_order = group_order or GROUP_ORDER
    notes = []
    for idx, (ax, cat) in enumerate(zip(axes, category_order)):
        sub = prop[prop[category_col] == cat]
        data, colors = [], []
        for g in group_order:
            data.append(sub[sub["group"] == g][value_col].to_numpy())
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
        for i, vals in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.14, 0.14, size=len(vals)), vals, s=1.4,
                       color=NEUTRAL_DARK, alpha=0.7, linewidths=0, zorder=3)
        non_empty = [d for d in data if len(d) > 0]
        star_txt = ""
        if len(non_empty) >= 2:
            p = stats.kruskal(*non_empty).pvalue
            star_txt = significance_stars(p) or "ns"
        title = label_map.get(cat, cat) if label_map else cat
        ax.set_title(f"{title}  {star_txt}".strip(), fontsize=5.0, pad=2.0)
        ax.set_xticks(range(1, len(group_order) + 1))
        ax.set_xticklabels([GROUP_LABELS[g] for g in group_order], rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.tick_params(length=1.2, pad=1.0)
        ax.margins(y=0.14)
        if idx % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=5.0, labelpad=1.5)
    for ax in axes[len(category_order):]:
        ax.set_visible(False)
    notes.append("Kruskal-Wallis across groups per category; "
                 "*** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    return notes


def _balance_modules(modules, present, nblocks):
    """Split modules into nblocks contiguous groups of roughly equal gene count."""
    items = [(name, [g for g in glist if g in present]) for name, glist in modules.items()]
    items = [(n, g) for n, g in items if g]
    total = sum(len(g) for _, g in items)
    target = total / nblocks
    blocks, current, count = [], [], 0
    for name, glist in items:
        if current and count + len(glist) / 2 > target * (len(blocks) + 1):
            blocks.append(current)
            current = []
        current.append((name, glist))
        count += len(glist)
    if current:
        blocks.append(current)
    while len(blocks) < nblocks:
        blocks.append([])
    return blocks[:nblocks]


def module_heatmap(fig, gs_area, cax, cells, modules, group_order=None,
                   module_colors=None, nblocks=2, gene_fontsize=5.0):
    """Group-level mean expression z-score per gene, split into balanced columns.

    Splitting keeps each column short enough that gene labels stay legible at the
    journal's minimum type size.
    """
    group_order = group_order or GROUP_ORDER
    module_colors = module_colors or {}
    present = set(cells.columns)

    all_genes, gene_module = [], {}
    for name, glist in modules.items():
        for g in glist:
            if g in present and g not in gene_module:
                all_genes.append(g)
                gene_module[g] = name

    mat = np.zeros((len(all_genes), len(group_order)))
    for j, grp in enumerate(group_order):
        sub = cells[cells["group"] == grp]
        for i, gene in enumerate(all_genes):
            mat[i, j] = sub[gene].mean()
    z = np.nan_to_num((mat - mat.mean(axis=1, keepdims=True))
                      / (mat.std(axis=1, keepdims=True) + 1e-9))
    vmax = float(np.abs(z).max())
    row_of = {g: i for i, g in enumerate(all_genes)}

    blocks = _balance_modules(modules, present, nblocks)
    heights = [sum(len(g) for _, g in b) for b in blocks]
    tallest = max(heights)

    inner = gs_area.subgridspec(1, nblocks, wspace=1.05)
    first_ax, im = None, None
    for bi, block in enumerate(blocks):
        block_genes = [g for _, glist in block for g in glist]
        cell = inner[bi].subgridspec(2, 2, width_ratios=[0.13, 1.0],
                                     height_ratios=[len(block_genes),
                                                    tallest - len(block_genes) + 1e-6],
                                     wspace=0.10, hspace=0.0)
        sax = fig.add_subplot(cell[0, 0])
        ax = fig.add_subplot(cell[0, 1])
        rows = [row_of[g] for g in block_genes]
        im = ax.imshow(z[rows, :], cmap=DIVERGING_CMAP, aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in group_order], rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        strip_axes(sax)
        for i, gene in enumerate(block_genes):
            sax.add_patch(_rect(0, i - 0.5, 1, 1,
                                module_colors.get(gene_module[gene], NEUTRAL_MID)))
            sax.text(-0.35, i, gene, ha="right", va="center", fontsize=gene_fontsize,
                     style="italic")
        sax.set_xlim(0, 1)
        sax.set_ylim(len(block_genes) - 0.5, -0.5)
        if first_ax is None:
            first_ax = sax

    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("z-score", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    return all_genes, first_ax


def _rect(x, y, w, h, color):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, color=color, linewidth=0)


def group_boxplots(axes, df, value_cols, group_col, group_order, labels=None,
                   pvalues=None, ylabel="Score", title_fontsize=5.0):
    """One small boxplot per measure, comparing a fixed set of groups."""
    rng = np.random.default_rng(7)
    for idx, (ax, col) in enumerate(zip(axes, value_cols)):
        data, colors = [], []
        for g in group_order:
            v = df.loc[df[group_col] == g, col].dropna().to_numpy()
            data.append(v)
            colors.append(GROUP_COLORS.get(g, NEUTRAL_MID))
        bp = ax.boxplot(data, widths=0.58, showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=0.7),
                        boxprops=dict(linewidth=0.3, edgecolor=NEUTRAL_DARK),
                        whiskerprops=dict(linewidth=0.3, color=NEUTRAL_DARK),
                        capprops=dict(linewidth=0.3, color=NEUTRAL_DARK))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for i, vals in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.13, 0.13, size=len(vals)), vals, s=2.0,
                       color=NEUTRAL_DARK, alpha=0.75, linewidths=0, zorder=3)
        star = ""
        if pvalues is not None and col in pvalues:
            p = pvalues[col]
            star = significance_stars(p) or "ns" if np.isfinite(p) else ""
        name = labels.get(col, col) if labels else col
        ax.set_title(f"{name}  {star}".strip(), fontsize=title_fontsize, pad=2.0)
        ax.set_xticks(range(1, len(group_order) + 1))
        ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in group_order], rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.tick_params(length=1.2, pad=1.0)
        ax.margins(y=0.16)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=5.0, labelpad=1.5)


def grouped_boxplots(ax, df, value_cols, group_col, group_order, labels=None,
                     pvalues=None, ylabel="Value", italic=False, rotate=40,
                     box_width=0.19):
    """All measures in one axes, with one box per group inside each measure.

    Preferred over one axes per measure when the panel is narrow: four rotated
    group labels under a half-inch axes collide, whereas a single shared legend
    does not.
    """
    rng = np.random.default_rng(7)
    n = len(group_order)
    offsets = (np.arange(n) - (n - 1) / 2) * box_width
    for i, col in enumerate(value_cols):
        for k, g in enumerate(group_order):
            v = df.loc[df[group_col] == g, col].dropna().to_numpy()
            if len(v) == 0:
                continue
            pos = i + offsets[k]
            bp = ax.boxplot([v], positions=[pos], widths=box_width * 0.86,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="white", linewidth=0.6),
                            boxprops=dict(linewidth=0.25, edgecolor=NEUTRAL_DARK),
                            whiskerprops=dict(linewidth=0.25, color=NEUTRAL_DARK),
                            capprops=dict(linewidth=0.25, color=NEUTRAL_DARK))
            bp["boxes"][0].set_facecolor(GROUP_COLORS.get(g, NEUTRAL_MID))
            bp["boxes"][0].set_alpha(0.85)
            ax.scatter(pos + rng.uniform(-box_width * 0.22, box_width * 0.22,
                                         size=len(v)), v, s=1.2,
                       color=NEUTRAL_DARK, alpha=0.7, linewidths=0, zorder=3)
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels([(labels or {}).get(c, c) for c in value_cols],
                       rotation=rotate, ha="right" if rotate else "center",
                       rotation_mode="anchor" if rotate else None,
                       style="italic" if italic else "normal")
    ax.set_xlim(-0.6, len(value_cols) - 0.4)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.grid(True, axis="y", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)
    if pvalues:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.13)
        top = ax.get_ylim()[1]
        for i, col in enumerate(value_cols):
            p = pvalues.get(col, np.nan)
            star = (significance_stars(p) or "ns") if np.isfinite(p) else ""
            ax.text(i, top, star, ha="center", va="top", fontsize=5.0,
                    color=NEUTRAL_DARK)


def nes_barplot(ax, df, label_col, value_col, pos_color, neg_color, xlabel="NES",
                max_label_chars=44):
    """Horizontal signed-effect bars, largest magnitude at the top."""
    d = df.copy().sort_values(value_col)
    labels = [_shorten(s, max_label_chars) for s in d[label_col]]
    y = np.arange(len(d))
    colors = [pos_color if v > 0 else neg_color for v in d[value_col]]
    ax.barh(y, d[value_col], color=colors, height=0.72, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.axvline(0, color=NEUTRAL_DARK, linewidth=0.4)
    ax.tick_params(length=1.2, pad=1.0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)


def paired_nes_bars(ax, df, label_col, cols, colors, names, xlabel="NES",
                    max_label_chars=44):
    """Two signed bars per row so opposing contrasts can be read against each other."""
    d = df.copy()
    d["_span"] = d[cols[0]] - d[cols[1]]
    d = d.sort_values("_span")
    y = np.arange(len(d))
    h = 0.36
    for k, (col, color, name) in enumerate(zip(cols, colors, names)):
        ax.barh(y + (0.5 - k) * h, d[col], height=h, color=color, linewidth=0,
                label=name)
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(s, max_label_chars) for s in d[label_col]])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.axvline(0, color=NEUTRAL_DARK, linewidth=0.4)
    ax.tick_params(length=1.2, pad=1.0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)


def count_barplot(ax, labels, values, colors, xlabel="Genes"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.65, len(labels) - 0.35)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, f"{v:,}", va="center", ha="left",
                fontsize=5.0, color=NEUTRAL_DARK)
    ax.set_xlim(0, max(values) * 1.16)


def trend_lines(ax, df, label_col, value_cols, x_labels, class_col=None,
                class_colors=None, n_label=3, ylabel="Mean expression"):
    """Per-pathway trajectories across an ordered set of groups.

    Lines are coloured by direction rather than by identity: 20 individually
    coloured lines cannot be matched to a legend at this size, whereas the
    direction split is the point of the panel.
    """
    x = np.arange(len(value_cols))
    labelled = []
    for _, row in df.iterrows():
        cls = row[class_col] if class_col else "other"
        color = (class_colors or {}).get(cls, NEUTRAL_MID)
        ax.plot(x, [row[c] for c in value_cols], color=color, linewidth=0.5,
                alpha=0.8, solid_capstyle="round")
    if class_col:
        for cls in df[class_col].unique():
            sub = df[df[class_col] == cls]
            span = (sub[value_cols[-1]] - sub[value_cols[0]]).abs().sort_values(
                ascending=False)
            for i in span.index[:n_label]:
                labelled.append([df.loc[i, label_col], float(df.loc[i, value_cols[-1]]),
                                 (class_colors or {}).get(cls, NEUTRAL_MID)])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.12, len(value_cols) - 1 + 0.06)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)

    # Nudge end labels apart vertically so overlapping trajectories stay readable.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    labelled.sort(key=lambda r: r[1])
    for _ in range(120):
        moved = False
        for i in range(len(labelled) - 1):
            delta = labelled[i + 1][1] - labelled[i][1]
            if delta < gap:
                shift = (gap - delta) / 2
                labelled[i][1] -= shift
                labelled[i + 1][1] += shift
                moved = True
        if not moved:
            break
    for name, yv, color in labelled:
        ax.annotate(_shorten(name, 24), xy=(x[-1], yv), xytext=(3.0, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=5.0, color=color, annotation_clip=False)


def zscore_heatmap(fig, gs_area, cax, matrix, row_labels, col_labels, blocks,
                   block_titles=None, gene_fontsize=5.0, wspace=1.15, italic=True):
    """Split a genes-by-groups z-score matrix into labelled column blocks."""
    vmax = float(np.abs(matrix).max())
    inner = gs_area.subgridspec(1, len(blocks), wspace=wspace)
    tallest = max(len(b) for b in blocks)
    first_ax, im = None, None
    for bi, rows in enumerate(blocks):
        cell = inner[bi].subgridspec(2, 1, height_ratios=[len(rows),
                                                          tallest - len(rows) + 1e-6],
                                     hspace=0.0)
        ax = fig.add_subplot(cell[0])
        im = ax.imshow(matrix[rows, :], cmap=DIVERGING_CMAP, aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row_labels[r] for r in rows],
                           style="italic" if italic else "normal",
                           fontsize=gene_fontsize)
        ax.tick_params(length=0, pad=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if block_titles:
            ax.set_title(block_titles[bi], fontsize=5.2, pad=2.0)
        if first_ax is None:
            first_ax = ax
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("z-score", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    return first_ax


def pseudotime_lines(ax, df, gene_col, x_col, y_col, color, n_label=3,
                     xlabel="Pseudotime bin", ylabel="Mean expression",
                     label_chars=9):
    """One thin line per gene plus their mean, with a few genes labelled.

    Ten individually coloured lines cannot be matched back to a legend at this
    size, so identity is carried by end labels and the shared trend by the mean.
    """
    genes = list(dict.fromkeys(df[gene_col]))
    ends = []
    for g in genes:
        sub = df[df[gene_col] == g].sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], color=color, linewidth=0.4, alpha=0.55)
        ends.append([g, float(sub[y_col].iloc[-1]), float(sub[x_col].iloc[-1])])
    mean_curve = df.groupby(x_col)[y_col].mean()
    ax.plot(mean_curve.index, mean_curve.to_numpy(), color=color, linewidth=1.2)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)

    ends.sort(key=lambda r: -abs(r[1] - df[df[gene_col] == r[0]][y_col].iloc[0]))
    chosen = ends[:n_label]
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.085
    chosen.sort(key=lambda r: r[1])
    for _ in range(80):
        moved = False
        for i in range(len(chosen) - 1):
            delta = chosen[i + 1][1] - chosen[i][1]
            if delta < gap:
                shift = (gap - delta) / 2
                chosen[i][1] -= shift
                chosen[i + 1][1] += shift
                moved = True
        if not moved:
            break
    for g, yv, xv in chosen:
        ax.annotate(_shorten(g, label_chars), xy=(xv, yv), xytext=(2.0, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=5.0, style="italic", color=color, annotation_clip=False)


def kegg_dotplot(fig, ax, cax, lax, frames, column_names, n_top=10,
                 label_col="Description", ratio_col="GeneRatio",
                 count_col="Count", padj_col="p.adjust", max_label_chars=38):
    """Enrichment dot plot with one column per gene set, dot size = gene count."""
    keep: list[str] = []
    for df in frames:
        top = df.sort_values(padj_col).head(n_top)
        for name in top[label_col]:
            if name not in keep:
                keep.append(name)

    def ratio(series):
        return series.map(lambda s: eval_ratio(s))

    size, color = np.full((len(keep), len(frames)), np.nan), np.full(
        (len(keep), len(frames)), np.nan)
    ratios = np.full((len(keep), len(frames)), np.nan)
    for j, df in enumerate(frames):
        d = df.set_index(label_col)
        for i, name in enumerate(keep):
            if name in d.index:
                row = d.loc[name]
                size[i, j] = float(row[count_col])
                color[i, j] = -np.log10(float(row[padj_col]))
                ratios[i, j] = eval_ratio(row[ratio_col])

    xs, ys = np.meshgrid(np.arange(len(frames)), np.arange(len(keep)))
    mask = ~np.isnan(size)
    sc = ax.scatter(xs[mask], ys[mask], s=size[mask] * 2.6, c=color[mask],
                    cmap=EXPR_CMAP, linewidths=0.15, edgecolors="white")
    ax.set_xticks(np.arange(len(frames)))
    ax.set_xticklabels(column_names)
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels([_shorten(s, max_label_chars) for s in keep])
    ax.set_xlim(-0.6, len(frames) - 0.4)
    ax.set_ylim(-0.7, len(keep) - 0.3)
    ax.invert_yaxis()
    ax.tick_params(length=0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)

    cb = fig.colorbar(sc, cax=cax)
    # Plain text rather than mathtext: subscripts render at 0.7x and would fall
    # below the 5 pt floor.
    cb.ax.set_title("\u2212log10\nadj. p", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)

    strip_axes(lax)
    counts = np.array([3, 6, 9])
    handles = [Line2D([], [], marker="o", linestyle="none",
                      markersize=np.sqrt(c * 2.6), markerfacecolor=NEUTRAL_MID,
                      markeredgecolor="none", label=str(c)) for c in counts]
    leg = lax.legend(handles=handles, title="Genes", loc="center left",
                     handletextpad=0.5, labelspacing=0.6, borderpad=0.2, fontsize=5.0)
    leg.get_title().set_fontsize(5.0)
    return keep


def eval_ratio(value):
    if isinstance(value, str) and "/" in value:
        a, b = value.split("/")
        return float(a) / float(b)
    return float(value)


def segment_heatmap(fig, ax, cax, matrix, row_labels, segments, segment_colors,
                    sax=None, title=None, segment_labels=None):
    """Heatmap over ordered columns that belong to labelled contiguous segments."""
    vmax = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap=DIVERGING_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, style="italic")
    ax.set_xticks([])
    ax.tick_params(length=0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bounds = []
    start = 0
    for k in range(1, len(segments) + 1):
        if k == len(segments) or segments[k] != segments[start]:
            bounds.append((start, k, segments[start]))
            start = k
    for a, b, name in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=1.2)
    if sax is not None:
        strip_axes(sax)
        for a, b, name in bounds:
            sax.add_patch(_rect(a - 0.5, 0, b - a, 1, segment_colors[name]))
            shown = (segment_labels or {}).get(name, name)
            sax.text((a + b - 1) / 2, 0.5, shown, ha="center", va="center",
                     fontsize=5.0, color="white", fontweight="bold")
        sax.set_xlim(-0.5, len(segments) - 0.5)
        sax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=5.4, pad=2.0)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("z-score", fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)


def grouped_bars(ax, df, cat_col, series_col, value_col, cat_order, series_order,
                 colors, labels=None, value_fmt=None, ylabel="", bar_width=0.38,
                 rotate=40):
    """Side-by-side bars: one cluster per category, one bar per series."""
    x = np.arange(len(cat_order))
    peak = 0.0
    for k, s in enumerate(series_order):
        sub = df[df[series_col] == s].set_index(cat_col)
        vals = [float(sub[value_col].get(c, 0.0)) for c in cat_order]
        peak = max(peak, max(vals) if vals else 0.0)
        offset = (k - (len(series_order) - 1) / 2) * bar_width
        ax.bar(x + offset, vals, width=bar_width, color=colors[s], linewidth=0,
               label=s)
        if value_fmt:
            for xi, v in zip(x + offset, vals):
                if v > 0:
                    ax.text(xi, v, value_fmt(v), ha="center", va="bottom",
                            fontsize=5.0, color=NEUTRAL_DARK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([(labels or {}).get(c, c) for c in cat_order],
                       rotation=rotate, ha="right" if rotate else "center",
                       rotation_mode="anchor" if rotate else None)
    ax.set_xlim(-0.6, len(cat_order) - 0.4)
    ax.set_ylim(0, peak * 1.45 if peak > 0 else 1)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.grid(True, axis="y", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)


def grouped_hbars(ax, df, cat_col, series_col, value_col, cat_order, series_order,
                  colors, xlabel="", bar_height=0.26, max_label_chars=30):
    """Horizontal grouped bars, one cluster per category, one bar per series.

    Horizontal because the categories here are pathway names, which do not fit
    under a tick at the journal's minimum type size.
    """
    y = np.arange(len(cat_order))
    for k, s in enumerate(series_order):
        sub = df[df[series_col] == s].set_index(cat_col)
        vals = [float(sub[value_col].get(c, 0.0)) for c in cat_order]
        offset = ((len(series_order) - 1) / 2 - k) * bar_height
        ax.barh(y + offset, vals, height=bar_height, color=colors[s], linewidth=0,
                label=s)
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(c, max_label_chars) for c in cat_order])
    ax.set_ylim(len(cat_order) - 0.45, -0.55)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)


def dumbbell(ax, df, label_col, col_a, col_b, colors, names, star_col=None,
             xlabel="Module score (median)", max_label_chars=16):
    """Paired medians joined by a rule, so the direction of change reads first."""
    d = df.copy().reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    for yi, (_, row) in zip(y, d.iterrows()):
        ax.plot([row[col_a], row[col_b]], [yi, yi], color="#C8C8C8", linewidth=0.8,
                zorder=1, solid_capstyle="round")
    ax.scatter(d[col_a], y, s=9, color=colors[0], linewidths=0, zorder=3,
               label=names[0])
    ax.scatter(d[col_b], y, s=9, color=colors[1], linewidths=0, zorder=3,
               label=names[1])
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(s, max_label_chars) for s in d[label_col]])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)
    if star_col:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.16)
        for yi, (_, row) in zip(y, d.iterrows()):
            ax.text(ax.get_xlim()[1], yi, row[star_col], ha="right", va="center",
                    fontsize=5.0, color=NEUTRAL_DARK)


def volcano(ax, df, label_genes, lfc_col="avg_log2FC", p_col="p_val_adj",
            gene_col="gene", padj_cut=0.05, lfc_cut=0.5, up_color="#C0392B",
            down_color="#4C7FB8", p_floor=1e-300):
    """Effect size against adjusted significance, with a fixed gene panel labelled."""
    d = df.dropna(subset=[lfc_col, p_col]).copy()
    d["_y"] = -np.log10(np.clip(d[p_col].to_numpy(), p_floor, None))
    sig = d[p_col] < padj_cut
    up = sig & (d[lfc_col] >= lfc_cut)
    down = sig & (d[lfc_col] <= -lfc_cut)
    rest = ~(up | down)
    ax.scatter(d.loc[rest, lfc_col], d.loc[rest, "_y"], s=1.2, color="#D5D5D5",
               linewidths=0, rasterized=True)
    ax.scatter(d.loc[down, lfc_col], d.loc[down, "_y"], s=2.2, color=down_color,
               linewidths=0, rasterized=True)
    ax.scatter(d.loc[up, lfc_col], d.loc[up, "_y"], s=2.2, color=up_color,
               linewidths=0, rasterized=True)
    ax.axhline(-np.log10(padj_cut), color=NEUTRAL_MID, linewidth=0.35, linestyle=(0, (3, 2)))
    for v in (-lfc_cut, lfc_cut):
        ax.axvline(v, color=NEUTRAL_MID, linewidth=0.35, linestyle=(0, (3, 2)))
    ax.set_xlabel("log2 fold change (EBV+ vs EBV\u2212)", labelpad=1.5)
    ax.set_ylabel("\u2212log10 adjusted p", labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)

    marked = d[d[gene_col].isin(label_genes)]
    if len(marked):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        anchors = np.column_stack([marked[lfc_col].to_numpy(),
                                   marked["_y"].to_numpy()])
        unit = np.column_stack([(anchors[:, 0] - x0) / (x1 - x0),
                                (anchors[:, 1] - y0) / (y1 - y0)])
        placed = repel(unit, min_dist=0.115)
        placed[:, 0] = np.clip(placed[:, 0], 0.02, 0.98)
        placed[:, 1] = np.clip(placed[:, 1], 0.02, 0.98)
        for (_, row), anchor, pos in zip(marked.iterrows(), anchors, placed):
            color = up_color if row[lfc_col] > 0 else down_color
            px = x0 + pos[0] * (x1 - x0)
            py = y0 + pos[1] * (y1 - y0)
            ax.plot([anchor[0], px], [anchor[1], py], color=color, linewidth=0.25,
                    zorder=3)
            ax.text(px, py, row[gene_col], fontsize=5.0, style="italic",
                    color=color, ha="center", va="center", zorder=5)
            ax.scatter([anchor[0]], [anchor[1]], s=6, facecolors="none",
                       edgecolors=color, linewidths=0.4, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    return int(up.sum()), int(down.sum())


def _split_runs(row_groups, nblocks):
    """Cut an ordered row list into nblocks chunks, preferring group boundaries."""
    runs, start = [], 0
    for k in range(1, len(row_groups) + 1):
        if k == len(row_groups) or row_groups[k] != row_groups[start]:
            runs.append(list(range(start, k)))
            start = k
    target = len(row_groups) / nblocks
    blocks, current, placed = [], [], 0
    for run in runs:
        if current and placed + len(current) + len(run) / 2 > target * (len(blocks) + 1):
            blocks.append(current)
            placed += len(current)
            current = []
        current.extend(run)
    if current:
        blocks.append(current)
    while len(blocks) < nblocks:
        blocks.append([])
    return blocks[:nblocks]


def annotated_block_heatmap(fig, gs_area, cax, matrix, row_labels, col_labels,
                            row_groups, group_colors, nblocks=2,
                            label_fontsize=5.0, italic=True, wspace=1.1,
                            cbar_title="z-score", vmax=None, col_rotation=40):
    """Row-annotated heatmap split into balanced column blocks.

    Rows must already be ordered by their annotation group; the split follows
    group boundaries so the colour strip stays contiguous.
    """
    vmax = float(np.nanmax(np.abs(matrix))) if vmax is None else vmax
    blocks = _split_runs(list(row_groups), nblocks)
    tallest = max(len(b) for b in blocks)
    inner = gs_area.subgridspec(1, nblocks, wspace=wspace)
    first_ax, im = None, None
    for bi, rows in enumerate(blocks):
        if not rows:
            continue
        cell = inner[bi].subgridspec(2, 2, width_ratios=[0.13, 1.0],
                                     height_ratios=[len(rows),
                                                    tallest - len(rows) + 1e-6],
                                     wspace=0.10, hspace=0.0)
        sax = fig.add_subplot(cell[0, 0])
        ax = fig.add_subplot(cell[0, 1])
        im = ax.imshow(matrix[rows, :], cmap=DIVERGING_CMAP, aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=col_rotation,
                           ha="right" if col_rotation else "center",
                           rotation_mode="anchor" if col_rotation else None)
        ax.set_yticks([])
        ax.tick_params(length=0, pad=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        strip_axes(sax)
        for i, r in enumerate(rows):
            sax.add_patch(_rect(0, i - 0.5, 1, 1,
                                group_colors.get(row_groups[r], NEUTRAL_MID)))
            sax.text(-0.35, i, row_labels[r], ha="right", va="center",
                     fontsize=label_fontsize, style="italic" if italic else "normal")
        sax.set_xlim(0, 1)
        sax.set_ylim(len(rows) - 0.5, -0.5)
        if first_ax is None:
            first_ax = sax
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title(cbar_title, fontsize=5.0, pad=2.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, width=0.4)
    cb.outline.set_linewidth(0.4)
    return first_ax


def split_violin(ax, df, cat_col, value_col, status_col, order, statuses, colors,
                 labels=None, ylabel="Normalized expression", italic=True,
                 stars=None, width=0.86, rotate=40):
    """Two half-violins per category so a paired contrast reads at a glance.

    A shared y axis across the categories is deliberate: free per-gene scaling
    would make a gene expressed at 0.1 look like one expressed at 5.
    """
    for i, cat in enumerate(order):
        for sign, status in zip((-1, 1), statuses):
            vals = df.loc[(df[cat_col] == cat) & (df[status_col] == status),
                          value_col].dropna().to_numpy()
            if len(vals) < 3 or np.allclose(vals, vals[0]):
                continue
            parts = ax.violinplot([vals], positions=[i], widths=width,
                                  showextrema=False)
            body = parts["bodies"][0]
            verts = body.get_paths()[0].vertices
            if sign < 0:
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, i)
            else:
                verts[:, 0] = np.clip(verts[:, 0], i, np.inf)
            body.set_facecolor(colors[status])
            body.set_alpha(0.85)
            body.set_linewidth(0)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([i + sign * 0.10] * 2, [q1, q3], color=NEUTRAL_DARK,
                    linewidth=0.5, solid_capstyle="butt", zorder=4)
            ax.plot([i + sign * 0.10], [med], marker="o", markersize=1.1,
                    color="white", markeredgecolor=NEUTRAL_DARK,
                    markeredgewidth=0.25, zorder=5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([(labels or {}).get(c, c) for c in order],
                       rotation=rotate, ha="right" if rotate else "center",
                       rotation_mode="anchor" if rotate else None,
                       style="italic" if italic else "normal")
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(length=1.2, pad=1.0)
    ax.grid(True, axis="y", color="#EDEDED", linewidth=0.3)
    ax.set_axisbelow(True)
    if stars:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.12)
        top = ax.get_ylim()[1]
        for i, cat in enumerate(order):
            ax.text(i, top, stars.get(cat, ""), ha="center", va="top",
                    fontsize=5.0, color=NEUTRAL_DARK)


def _shorten(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"
