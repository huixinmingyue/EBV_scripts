#!/usr/bin/env Rscript
# Data-only export for Figure 10: the B-cell monocle2 trajectory (DDRTree
# coordinates, the fitted tree, state and pseudotime) and the per-state mean
# expression of the subtype markers. No plotting.

suppressPackageStartupMessages({
  library(monocle)
  library(Matrix)
  library(igraph)
  library(data.table)
})

rdata <- "F:/EBV/Rdata"
rdata_x <- "F:/EBV/EBV_xiaomi/EBV/Rdata"
out <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig10"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

keep_groups <- c("HC", "IM", "MH_CD4", "HLH")

findf <- function(...) {
  hit <- c(...)[file.exists(c(...))]
  if (!length(hit)) stop("Missing: ", paste(c(...), collapse = " | "))
  hit[[1]]
}

cat("loading B-cell CDS...\n")
cds <- readRDS(findf(
  file.path(rdata_x, "cds_bcells_new_celltype_monocle2_no_regress.rds"),
  file.path(rdata, "cds_bcells_new_celltype_monocle2_no_regress.rds")))
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
  celltype = as.character(pData(cds)$new_celltype),
  sample = as.character(pData(cds)$orig.ident)
)
fwrite(cells, file.path(out, "fig10_trajectory.csv"))

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
fwrite(edges, file.path(out, "fig10_tree_edges.csv"))
cat("tree edges:", nrow(edges), "\n")

# Which subtype dominates each state: the trajectory is unsupervised, so this is
# what ties the states back to the annotation used in the rest of the figure.
comp <- cells[, .N, by = .(State, celltype)]
comp[, fraction := N / sum(N), by = State]
fwrite(comp[order(State, -fraction)], file.path(out, "fig10_state_composition.csv"))

cat("=== DONE ===\n")
