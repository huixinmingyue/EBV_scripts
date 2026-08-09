#!/usr/bin/env Rscript
# Data-only export for Figure 1. No plotting libraries, no graphics devices.
# Emits per-cell UMAP coordinates, grouping metadata, and lineage-marker
# expression so the figure can be rendered entirely in Python.

suppressPackageStartupMessages({
  library(Seurat)
  library(data.table)
})

options(future.globals.maxSize = 32000 * 1024^2)

rds_path <- "F:/EBV/Rdata/immune.combined.processed.rds"
out_dir <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig1"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

KEEP_GROUPS <- c("HC", "IM", "MH_CD4", "HLH")
GENES <- c("PTPRC", "CD3D", "CD4", "CD8A", "NCAM1", "FCGR3A", "CD19", "CD14")

cat("=== loading object ===\n")
obj <- readRDS(rds_path)
cat("cells total:", ncol(obj), "\n")
cat("meta columns:", paste(colnames(obj@meta.data), collapse = ", "), "\n")
cat("reductions:", paste(Reductions(obj), collapse = ", "), "\n")

cat("\n=== group table (all) ===\n")
print(table(obj$Newgroup6, useNA = "ifany"))

keep_cells <- colnames(obj)[as.character(obj$Newgroup6) %in% KEEP_GROUPS]
obj <- subset(obj, cells = keep_cells)
cat("\ncells kept (", paste(KEEP_GROUPS, collapse = "/"), "):", ncol(obj), "\n")
print(table(obj$Newgroup6, useNA = "ifany"))

if (length(Layers(obj[["RNA"]])) > 1) obj <- JoinLayers(obj, assay = "RNA")

umap <- Embeddings(obj, "umap")
meta <- obj@meta.data

dt <- data.table(
  cell_id = colnames(obj),
  UMAP1 = as.numeric(umap[, 1]),
  UMAP2 = as.numeric(umap[, 2]),
  cell_type3 = as.character(meta$cell_type3),
  Newgroup6 = as.character(meta$Newgroup6),
  orig.ident = as.character(meta$orig.ident)
)

present <- GENES[GENES %in% rownames(obj)]
missing <- setdiff(GENES, present)
if (length(missing) > 0) cat("\nWARNING genes absent:", paste(missing, collapse = ", "), "\n")

expr <- GetAssayData(obj, assay = "RNA", layer = "data")[present, , drop = FALSE]
for (g in present) dt[[g]] <- as.numeric(expr[g, ])

fwrite(dt, file.path(out_dir, "fig1_cells.csv"))
cat("\nwrote fig1_cells.csv rows:", nrow(dt), "cols:", ncol(dt), "\n")

# Per-patient cell-type proportions, recomputed from the same kept cells so
# panel d matches panels a-c exactly.
counts <- dt[, .(count = .N), by = .(orig.ident, cell_type3, Newgroup6)]
totals <- dt[, .(total = .N), by = .(orig.ident)]
prop <- merge(counts, totals, by = "orig.ident")
prop[, proportion := count / total * 100]
fwrite(prop, file.path(out_dir, "fig1_patient_proportions.csv"))
cat("wrote fig1_patient_proportions.csv rows:", nrow(prop), "\n")

cat("\ncell_type3 levels by size:\n")
print(sort(table(dt$cell_type3), decreasing = TRUE))

cat("\n=== DONE ===\n")
