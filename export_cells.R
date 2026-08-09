#!/usr/bin/env Rscript
# Generic data-only exporter for Seurat objects. No plotting libraries and no
# graphics devices: this only converts an .rds into CSV so that every figure can
# be rendered in Python.
#
# Usage:
#   Rscript export_cells.R --rds <path> --out <dir> --tag <name>
#                          [--groups HC,IM,MH_CD4,HLH]
#                          [--celltype-col celltype_new]
#                          [--group-col Newgroup6]
#                          [--genes <file with one gene per line>]
#                          [--extra-meta col1,col2]

suppressPackageStartupMessages({
  library(Seurat)
  library(data.table)
})

options(future.globals.maxSize = 64000 * 1024^2)

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  args[i + 1]
}

rds_path <- get_arg("--rds")
out_dir <- get_arg("--out")
tag <- get_arg("--tag", "cells")
group_col <- get_arg("--group-col", "Newgroup6")
celltype_col <- get_arg("--celltype-col", "celltype_new")
genes_file <- get_arg("--genes")
groups_arg <- get_arg("--groups")
extra_meta <- get_arg("--extra-meta")

stopifnot(!is.null(rds_path), !is.null(out_dir))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== loading", rds_path, "===\n")
obj <- readRDS(rds_path)
cat("cells total:", ncol(obj), "\n")
cat("meta columns:", paste(colnames(obj@meta.data), collapse = ", "), "\n")
cat("reductions:", paste(Reductions(obj), collapse = ", "), "\n")

if (!group_col %in% colnames(obj@meta.data)) stop("missing group column: ", group_col)
if (!celltype_col %in% colnames(obj@meta.data)) stop("missing celltype column: ", celltype_col)

cat("\n=== group table (all) ===\n")
print(table(obj@meta.data[[group_col]], useNA = "ifany"))
cat("\n=== celltype table (all) ===\n")
print(table(obj@meta.data[[celltype_col]], useNA = "ifany"))

if (!is.null(groups_arg)) {
  keep_groups <- strsplit(groups_arg, ",")[[1]]
  keep_cells <- colnames(obj)[as.character(obj@meta.data[[group_col]]) %in% keep_groups]
  obj <- subset(obj, cells = keep_cells)
  cat("\ncells kept (", paste(keep_groups, collapse = "/"), "):", ncol(obj), "\n")
}

# Joining layers is only needed to read expression; skip it for metadata-only
# exports so very large objects stay cheap.
if (!is.null(genes_file) && length(Layers(obj[["RNA"]])) > 1) {
  obj <- JoinLayers(obj, assay = "RNA")
}

red <- if ("umap_integrated" %in% Reductions(obj)) "umap_integrated" else
       if ("umap" %in% Reductions(obj)) "umap" else Reductions(obj)[1]
cat("using reduction:", red, "\n")
emb <- Embeddings(obj, red)
meta <- obj@meta.data

dt <- data.table(
  cell_id = colnames(obj),
  UMAP1 = as.numeric(emb[, 1]),
  UMAP2 = as.numeric(emb[, 2]),
  celltype = as.character(meta[[celltype_col]]),
  group = as.character(meta[[group_col]]),
  orig.ident = as.character(meta$orig.ident)
)

if (!is.null(extra_meta)) {
  for (col in strsplit(extra_meta, ",")[[1]]) {
    if (col %in% colnames(meta)) {
      dt[[col]] <- as.character(meta[[col]])
    } else {
      cat("WARNING extra meta column absent:", col, "\n")
    }
  }
}

if (!is.null(genes_file)) {
  genes <- readLines(genes_file, warn = FALSE)
  genes <- trimws(genes)
  genes <- unique(genes[nzchar(genes)])
  present <- genes[genes %in% rownames(obj)]
  missing <- setdiff(genes, present)
  if (length(missing) > 0) cat("WARNING genes absent:", paste(missing, collapse = ", "), "\n")
  cat("exporting", length(present), "of", length(genes), "genes\n")
  expr <- GetAssayData(obj, assay = "RNA", layer = "data")[present, , drop = FALSE]
  for (g in present) dt[[g]] <- as.numeric(expr[g, ])
  writeLines(present, file.path(out_dir, paste0(tag, "_genes_present.txt")))
}

fwrite(dt, file.path(out_dir, paste0(tag, "_cells.csv")))
cat("wrote", paste0(tag, "_cells.csv"), "rows:", nrow(dt), "cols:", ncol(dt), "\n")

counts <- dt[, .(count = .N), by = .(orig.ident, celltype, group)]
totals <- dt[, .(total = .N), by = .(orig.ident)]
prop <- merge(counts, totals, by = "orig.ident")
prop[, proportion := count / total * 100]
fwrite(prop, file.path(out_dir, paste0(tag, "_patient_proportions.csv")))
cat("wrote", paste0(tag, "_patient_proportions.csv"), "rows:", nrow(prop), "\n")

cat("\n=== DONE ===\n")
