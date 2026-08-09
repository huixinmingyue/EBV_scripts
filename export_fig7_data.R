#!/usr/bin/env Rscript
# Data-only export for Figure 7: the NK monocle2 trajectory (DDRTree
# coordinates, the fitted tree, state and pseudotime) plus the per-state mean
# expression of the state-defining markers. No plotting.

suppressPackageStartupMessages({
  library(monocle)
  library(Matrix)
  library(igraph)
  library(data.table)
})

rdata <- "F:/EBV/Rdata"
rdata_x <- "F:/EBV/EBV_xiaomi/EBV/Rdata"
fig07 <- "F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig07"
out <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig7"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

keep_groups <- c("HC", "IM", "MH_CD4", "HLH")

findf <- function(...) {
  hit <- c(...)[file.exists(c(...))]
  if (!length(hit)) stop("Missing: ", paste(c(...), collapse = " | "))
  hit[[1]]
}

cat("loading NK CDS...\n")
cds <- readRDS(findf(
  file.path(rdata_x, "cds_merged_nk_new_integrated_new_celltype_monocle2_ds1300_no_regress.rds"),
  file.path(rdata, "cds_merged_nk_new_integrated_new_celltype_monocle2_ds1300_no_regress.rds")))
keep <- colnames(cds)[as.character(pData(cds)$Newgroup6) %in% keep_groups]
cds <- cds[, keep]
cat("cells:", ncol(cds), "\n")
print(table(pData(cds)$Newgroup6))
print(table(pData(cds)$State))

y <- reducedDimS(cds)
cells <- data.table(
  cell_id = colnames(cds),
  Component1 = as.numeric(y[1, colnames(cds)]),
  Component2 = as.numeric(y[2, colnames(cds)]),
  State = as.character(pData(cds)$State),
  group = as.character(pData(cds)$Newgroup6),
  Pseudotime = as.numeric(pData(cds)$Pseudotime),
  celltype = as.character(pData(cds)$new_celltype)
)
fwrite(cells, file.path(out, "fig7_trajectory.csv"))

edges <- data.table(x = numeric(), y = numeric(), xend = numeric(), yend = numeric())
mst <- tryCatch(minSpanningTree(cds), error = function(e) NULL)
kk <- tryCatch(reducedDimK(cds), error = function(e) NULL)
if (!is.null(mst) && !is.null(kk)) {
  vn <- colnames(kk)
  if (is.null(vn)) vn <- V(mst)$name
  vdf <- data.frame(vertex = vn, x = as.numeric(kk[1, seq_along(vn)]),
                    y = as.numeric(kk[2, seq_along(vn)]))
  ed <- igraph::as_data_frame(mst, what = "edges")
  fi <- match(ed$from, vdf$vertex)
  ti <- match(ed$to, vdf$vertex)
  ok <- is.finite(fi) & is.finite(ti)
  edges <- data.table(x = vdf$x[fi[ok]], y = vdf$y[fi[ok]],
                      xend = vdf$x[ti[ok]], yend = vdf$y[ti[ok]])
}
fwrite(edges, file.path(out, "fig7_tree_edges.csv"))
cat("tree edges:", nrow(edges), "\n")

# ---- per-state mean expression of the state-defining markers ---------------
mk <- read.csv(file.path(fig07, "panel_k_state_markers_noIMM.csv"),
               stringsAsFactors = FALSE)
gene_order <- unique(mk$gene)
expr <- log1p(Matrix::t(Matrix::t(exprs(cds)) / sizeFactors(cds)))
gene_order <- gene_order[gene_order %in% rownames(expr)]
state_vec <- as.character(pData(cds)$State)
states <- sort(unique(state_vec))
avg <- sapply(states, function(st)
  as.numeric(Matrix::rowMeans(expr[gene_order, state_vec == st, drop = FALSE])))
rownames(avg) <- gene_order
colnames(avg) <- states

long <- rbindlist(lapply(states, function(st) {
  data.table(gene = gene_order, state = st, mean_expr = avg[, st])
}))
long <- merge(long, data.table(gene = mk$gene, marker_state = as.character(mk$State)),
              by = "gene", all.x = TRUE)
fwrite(long, file.path(out, "fig7_state_marker_expr.csv"))
cat("state markers:", length(gene_order), "genes over", length(states), "states\n")

cat("=== DONE ===\n")
