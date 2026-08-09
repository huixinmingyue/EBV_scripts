#!/usr/bin/env Rscript
# Data-only export for Figure 8: the monocyte object's embedding, marker
# expression and per-patient proportions, plus the highly-variable-gene panel
# that separates MC_NLRP3_HK2 from the other subtypes. No plotting.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
})

options(future.globals.maxSize = 32000 * 1024^2)

out <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig8"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

rds <- if (file.exists("F:/EBV/EBV_xiaomi/EBV/Rdata/mcall_new_celltype.rds"))
  "F:/EBV/EBV_xiaomi/EBV/Rdata/mcall_new_celltype.rds" else
  "F:/EBV/Rdata/mcall_new_celltype.rds"

keep_groups <- c("HC", "IM", "MH_CD4", "HLH")
subtype_order <- c("MC_CD14_S100A8_CD163_RETN", "MC_NLRP3_HK2",
                   "MC_CD16_CX3CR1", "MC_SPIB_HLA-DRA_FLT3")
target_ct <- "MC_NLRP3_HK2"
markers <- c("CD14", "S100A8", "CD163", "RETN", "FCGR3A", "CX3CR1", "NLRP3",
             "HK2", "SPIB", "HLA-DRA", "FLT3", "KYNU", "IL1B", "CASP1", "TNF")

cat("loading", rds, "...\n")
obj <- readRDS(rds)
DefaultAssay(obj) <- "RNA"
if (length(Layers(obj[["RNA"]])) > 1) obj <- JoinLayers(obj, assay = "RNA")
obj <- subset(obj, cells = colnames(obj)[as.character(obj$Newgroup6) %in% keep_groups])
cat("cells:", ncol(obj), "\n")
print(table(obj$Newgroup6))
print(table(obj$new_celltype))

red <- if ("umap" %in% Reductions(obj)) "umap" else Reductions(obj)[1]
emb <- Embeddings(obj, red)
expr <- GetAssayData(obj, assay = "RNA", layer = "data")
present <- markers[markers %in% rownames(expr)]
cat("markers present:", paste(present, collapse = ", "), "\n")
cat("markers absent:", paste(setdiff(markers, present), collapse = ", "), "\n")

dt <- data.table(
  UMAP1 = as.numeric(emb[, 1]), UMAP2 = as.numeric(emb[, 2]),
  celltype = as.character(obj$new_celltype),
  group = as.character(obj$Newgroup6),
  sample = as.character(obj$orig.ident)
)
for (g in present) dt[[g]] <- as.numeric(expr[g, ])
fwrite(dt, file.path(out, "fig8_cells.csv"))

prop <- dt[, .(count = .N), by = .(sample, group, celltype)]
prop[, proportion := count / sum(count) * 100, by = sample]
fwrite(prop, file.path(out, "fig8_patient_proportions.csv"))

# ---- highly variable genes separating MC_NLRP3_HK2 ------------------------
is_valid <- function(g) !grepl("^ENSG|^RPL|^RPS|^MT-", g, ignore.case = TRUE)
ct <- as.character(obj$new_celltype)
tgt <- ct == target_ct
valid <- rownames(expr)[is_valid(rownames(expr))]
e_tgt <- expr[valid, tgt, drop = FALSE]
mu <- Matrix::rowMeans(e_tgt)
v <- pmax(Matrix::rowMeans(e_tgt^2) - mu^2, 0)
hvg <- names(sort(v, decreasing = TRUE))[seq_len(min(2000L, length(v)))]
mean_tgt <- Matrix::rowMeans(expr[hvg, tgt, drop = FALSE])
mean_other <- Matrix::rowMeans(expr[hvg, !tgt, drop = FALSE])
lfc <- log2((mean_tgt + 1e-4) / (mean_other + 1e-4))
ord <- order(lfc, decreasing = TRUE)
genes_plot <- c(hvg[ord][1:10], rev(hvg[ord])[1:10])
regulation <- c(rep("Up", 10), rep("Down", 10))
cat("HVG panel:", paste(genes_plot, collapse = ", "), "\n")

by_subtype <- rbindlist(lapply(subtype_order, function(s) {
  cells <- ct == s
  data.table(gene = genes_plot, subtype = s, regulation = regulation,
             mean_expr = as.numeric(Matrix::rowMeans(expr[genes_plot, cells, drop = FALSE])))
}))
fwrite(by_subtype, file.path(out, "fig8_hvg_by_subtype.csv"))

by_group <- rbindlist(lapply(keep_groups, function(g) {
  cells <- tgt & as.character(obj$Newgroup6) == g
  data.table(gene = genes_plot, group = g, regulation = regulation,
             mean_expr = as.numeric(Matrix::rowMeans(expr[genes_plot, cells, drop = FALSE])))
}))
fwrite(by_group, file.path(out, "fig8_hvg_by_group.csv"))

cat("=== DONE ===\n")
