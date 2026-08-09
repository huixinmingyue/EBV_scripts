#!/usr/bin/env Rscript
# Data-only export for Figure 6: per-cell expression and module scores for
# EBV+ versus EBV- HLH CD8+ T cells, in the exhausted subset and in monocle
# state 1. No plotting.

suppressPackageStartupMessages({
  library(data.table)
})

rdata <- "F:/EBV/EBV_xiaomi/EBV/Rdata"
out <- "F:/EBV/EBV_xiaomi/Figures_Nature_Python/_data/fig6"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

module_score <- function(genes, mat) {
  genes <- genes[genes %in% rownames(mat)]
  if (length(genes) == 0) return(rep(NA_real_, ncol(mat)))
  colMeans(mat[genes, , drop = FALSE])
}

long_expr <- function(mat, status, genes) {
  genes <- genes[genes %in% rownames(mat)]
  rbindlist(lapply(genes, function(g) {
    data.table(cell_id = colnames(mat), EBV_status = status, gene = g,
               expression = as.numeric(mat[g, ]))
  }))
}

write_sets <- function(sets, context) {
  rbindlist(lapply(names(sets), function(n) {
    data.table(context = context, set = n, gene = sets[[n]])
  }))
}

# ---- panels a-d: exhausted CD8+ T cells ------------------------------------
env1 <- new.env()
load(file.path(rdata, "cd8_exhausted_ebv_extracted.RData"), envir = env1)
cat("exhausted objects:", paste(ls(env1), collapse = ", "), "\n")
mat1 <- env1$expr_matrix
md1 <- env1$md
status1 <- as.character(md1$EBV_positive[match(colnames(mat1), rownames(md1))])
status1 <- ifelse(status1 == "EBV_positive", "EBV+", "EBV-")
cat("exhausted cells:", ncol(mat1), " EBV+:", sum(status1 == "EBV+"), "\n")

modules_exh <- list(
  Exhaustion = c("PDCD1", "LAG3", "TIGIT", "HAVCR2", "TOX"),
  Cytotoxic = c("IFNG", "GZMB", "PRF1", "NKG7", "CCL5"),
  Inflammation = c("TNF", "CXCL8", "IL1B", "CCL3"),
  Memory = c("IL7R", "TCF7", "LEF1"),
  Proliferation = c("MKI67", "TYMS", "TOP2A")
)
scores1 <- rbindlist(lapply(names(modules_exh), function(n) {
  data.table(cell_id = colnames(mat1), EBV_status = status1, module = n,
             score = module_score(modules_exh[[n]], mat1))
}))
fwrite(scores1, file.path(out, "fig6_exhausted_modules.csv"))
fwrite(long_expr(mat1, status1, env1$all_genes),
       file.path(out, "fig6_exhausted_expression.csv"))

sets1 <- list(exhaustion = env1$exhaustion_genes,
              inflammation = env1$inflammation_genes,
              cytotoxic = env1$cytotoxic_genes)
sets1 <- lapply(sets1, function(g) g[g %in% rownames(mat1)])
fwrite(write_sets(sets1, "exhausted"), file.path(out, "fig6_gene_sets_exhausted.csv"))
fwrite(write_sets(modules_exh, "exhausted_modules"),
       file.path(out, "fig6_module_members_exhausted.csv"))

rm(env1, mat1, md1)
invisible(gc())

# ---- panels e-g: monocle state 1 CD8+ T cells ------------------------------
env2 <- new.env()
load(file.path(rdata, "cd8_state1_ebv_expr.RData"), envir = env2)
cat("state1 objects:", paste(ls(env2), collapse = ", "), "\n")
mat2 <- env2$expr_matrix
md2 <- env2$state1_md[env2$hlh_state1_cells, ]
status2 <- as.character(md2$EBV_positive[match(colnames(mat2), rownames(md2))])
status2 <- ifelse(status2 == "EBV_positive", "EBV+", "EBV-")
cat("state1 cells:", ncol(mat2), " EBV+:", sum(status2 == "EBV+"), "\n")

sets2 <- list(
  `Cell cycle` = env2$cell_cycle_genes,
  `DNA damage repair` = env2$dna_damage_genes,
  `Interferon response` = env2$interferon_genes,
  Exhaustion = env2$exhaustion_genes,
  Cytotoxic = env2$cytotoxic_genes
)
sets2 <- lapply(sets2, function(g) g[g %in% rownames(mat2)])
scores2 <- rbindlist(lapply(names(sets2), function(n) {
  data.table(cell_id = colnames(mat2), EBV_status = status2, module = n,
             score = module_score(sets2[[n]], mat2))
}))
fwrite(scores2, file.path(out, "fig6_state1_modules.csv"))
fwrite(long_expr(mat2, status2, env2$all_genes),
       file.path(out, "fig6_state1_expression.csv"))
fwrite(write_sets(sets2, "state1"), file.path(out, "fig6_gene_sets_state1.csv"))

cat("=== DONE ===\n")
