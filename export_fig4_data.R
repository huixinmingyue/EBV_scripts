#!/usr/bin/env Rscript
# Data-only export for Figure 4. No plotting libraries, no graphics devices.
# Emits the DDRTree trajectory coordinates and the pseudotime-binned expression
# of the top homeostatic/progression genes so the figure renders in Python.

suppressPackageStartupMessages({
  library(monocle)
  library(Seurat)
  library(Matrix)
  library(data.table)
})

options(future.globals.maxSize = 64000 * 1024^2)

RDATA <- "F:/EBV/Rdata"
TC4 <- "F:/EBV/EBV_xiaomi/TC_EBV/04_monocle2_4groups/out"
REGEN4 <- "F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig04/TC_EBV_04_monocle2_reroot_HC"
OUT <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig4"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

N_BINS <- 20L
N_TOP <- 10L

cat("=== loading rerooted CDS ===\n")
cds <- readRDS(file.path(TC4, "cds_cd8_monocle2_reroot_HC.rds"))
cat("cds cells:", ncol(cds), "\n")

Y <- reducedDimS(cds)
pd <- as.data.frame(pData(cds))
traj <- data.table(
  cell = colnames(cds),
  Component1 = as.numeric(Y[1, ]),
  Component2 = as.numeric(Y[2, ]),
  State = as.character(pd$State),
  Pseudotime = as.numeric(pd$Pseudotime),
  Newgroup6 = as.character(pd$Newgroup6),
  celltype_new = as.character(pd$celltype_new),
  orig.ident = as.character(pd$orig.ident)
)
fwrite(traj, file.path(OUT, "fig4_trajectory.csv"))
cat("wrote fig4_trajectory.csv rows:", nrow(traj), "\n")
print(table(traj$Newgroup6, traj$State))

rm(cds); gc()

cat("\n=== pseudotime-binned expression for panel e ===\n")
home <- fread(file.path(REGEN4, "homeostatic_genes.csv"))
prog <- fread(file.path(REGEN4, "progression_genes.csv"))
home_top <- head(home[order(r)]$gene, N_TOP)
prog_top <- head(prog[order(-r)]$gene, N_TOP)
cat("homeostatic:", paste(home_top, collapse = ", "), "\n")
cat("progression:", paste(prog_top, collapse = ", "), "\n")

state_map <- fread(file.path(REGEN4, "cd8_state_pseudotime_map_4groups.csv"))
obj <- readRDS(file.path(RDATA, "merged_cd8_new_integrated_with_EBV.rds"))
if (length(Layers(obj[["RNA"]])) > 1) obj <- JoinLayers(obj, assay = "RNA")
common <- intersect(colnames(obj), state_map$cell)
cat("cells shared with trajectory:", length(common), "\n")
obj <- subset(obj, cells = common)

# Match the upstream normalisation exactly: log1p counts-per-10k.
counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
cs <- Matrix::colSums(counts)
expr <- log1p(Matrix::t(Matrix::t(counts) / (cs / 1e4)))

genes <- unique(c(home_top, prog_top))
genes <- genes[genes %in% rownames(expr)]
pt <- setNames(state_map$Pseudotime, state_map$cell)[colnames(obj)]
ord <- order(pt)
cells_sorted <- colnames(obj)[ord]
bin_id <- rep(seq_len(N_BINS), each = ceiling(length(cells_sorted) / N_BINS))[
  seq_len(length(cells_sorted))]

rows <- lapply(genes, function(g) {
  x <- as.numeric(expr[g, cells_sorted])
  data.table(
    gene = g,
    bin = as.integer(names(tapply(x, bin_id, mean))),
    mean_expr = as.numeric(tapply(x, bin_id, mean)),
    mean_pseudotime = as.numeric(tapply(pt[cells_sorted], bin_id, mean)),
    class = ifelse(g %in% home_top, "Homeostatic", "Progression")
  )
})
binned <- rbindlist(rows)
fwrite(binned, file.path(OUT, "fig4_pseudotime_binned_expr.csv"))
cat("wrote fig4_pseudotime_binned_expr.csv rows:", nrow(binned),
    "genes:", length(genes), "bins:", N_BINS, "\n")

cat("\n=== DONE ===\n")
