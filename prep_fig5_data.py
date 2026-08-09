"""Assemble the Figure 5 inputs from already-exported CSVs.

The EBV calls live in flat metadata tables while the embeddings live in the
per-figure exports, so the only real work here is a barcode-level join. No
Seurat object is opened.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python")
DATA = ROOT / "_data"
OUT = DATA / "fig5"

RDATA = Path("F:/EBV/Rdata")
RDATA_X = Path("F:/EBV/EBV_xiaomi/EBV/Rdata")

GROUPS = ["HC", "IM", "MH_CD4", "HLH"]


def barcode(series: pd.Series) -> pd.Series:
    return series.str.rsplit("_", n=1).str[-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- immune compartment: embedding + EBV call --------------------------
    umap = pd.read_csv(DATA / "fig1" / "fig1_cells.csv",
                       usecols=["cell_id", "UMAP1", "UMAP2", "cell_type3",
                                "Newgroup6", "orig.ident"])
    meta = pd.read_csv(RDATA / "immune_metadata_merged_with_HV4.csv",
                       usecols=["cell_id", "EBV_positive", "cell_type3",
                                "Newgroup6", "orig.ident_fixed"])
    umap["key"] = umap["orig.ident"] + "|" + barcode(umap["cell_id"])
    meta["key"] = meta["orig.ident_fixed"] + "|" + barcode(meta["cell_id"])
    dup = meta["key"].duplicated().sum()
    call = meta.drop_duplicates("key").set_index("key")["EBV_positive"]
    umap["hit"] = umap["key"].isin(call.index)
    # The HV4 EBV assay covers only the original capture batch; later samples
    # have no immune-wide call at all, so they are excluded rather than being
    # silently counted as EBV-negative.
    coverage = umap.groupby("orig.ident")["hit"].mean()
    covered = sorted(coverage[coverage >= 0.9].index)
    dropped = sorted(coverage[coverage < 0.9].index)
    umap = umap[umap["orig.ident"].isin(covered) & umap["hit"]].copy()
    umap["ebv"] = umap["key"].map(call).eq("EBV_positive")
    print(f"immune join: {len(umap):,} cells with an EBV call, "
          f"{dup} duplicate keys dropped, {int(umap['ebv'].sum())} EBV+")
    print(f"  samples kept ({len(covered)}): {covered}")
    print(f"  samples without HV4 coverage ({len(dropped)}): {dropped}")
    umap.rename(columns={"cell_type3": "celltype", "Newgroup6": "group"}, inplace=True)
    umap[["UMAP1", "UMAP2", "celltype", "group", "ebv"]].to_csv(
        OUT / "fig5_immune_umap.csv", index=False)
    pd.Series(covered, name="sample").to_csv(OUT / "fig5_immune_samples.csv",
                                             index=False)

    # per-sample EBV+ rate by lineage, for the group bar and lineage boxplots
    meta = meta[meta["Newgroup6"].isin(GROUPS)
                & meta["orig.ident_fixed"].isin(covered)].copy()
    meta["pos"] = meta["EBV_positive"].eq("EBV_positive")
    lineage = (meta.groupby(["orig.ident_fixed", "Newgroup6", "cell_type3"])
               .agg(n=("pos", "size"), n_pos=("pos", "sum")).reset_index())
    lineage["rate"] = lineage["n_pos"] / lineage["n"] * 100
    lineage.rename(columns={"orig.ident_fixed": "sample", "Newgroup6": "group",
                            "cell_type3": "celltype"}, inplace=True)
    lineage.to_csv(OUT / "fig5_immune_sample_lineage.csv", index=False)

    sample = (meta.groupby(["orig.ident_fixed", "Newgroup6"])
              .agg(n=("pos", "size"), n_pos=("pos", "sum")).reset_index())
    sample["rate"] = sample["n_pos"] / sample["n"] * 100
    sample.rename(columns={"orig.ident_fixed": "sample", "Newgroup6": "group"},
                  inplace=True)
    sample.to_csv(OUT / "fig5_immune_sample_rate.csv", index=False)
    print(f"immune per-sample rates: {len(sample)} samples over {GROUPS}")

    # ---- CD8 compartment: embedding already carries the EBV call ------------
    cd8 = pd.read_csv(DATA / "fig3" / "fig3_cells.csv",
                      usecols=["UMAP1", "UMAP2", "celltype", "group",
                               "orig.ident", "EBV_positive"])
    cd8["ebv"] = cd8["EBV_positive"].eq("EBV_positive")
    cd8[["UMAP1", "UMAP2", "celltype", "group", "ebv"]].to_csv(
        OUT / "fig5_cd8_umap.csv", index=False)
    print(f"CD8: {len(cd8):,} cells, {int(cd8['ebv'].sum())} EBV+")

    sub = (cd8.groupby(["orig.ident", "group", "celltype"])
           .agg(n=("ebv", "size"), n_pos=("ebv", "sum")).reset_index())
    sub["rate"] = sub["n_pos"] / sub["n"] * 100
    sub.rename(columns={"orig.ident": "sample"}, inplace=True)
    sub.to_csv(OUT / "fig5_cd8_sample_subtype.csv", index=False)

    # ---- tables copied verbatim so the figure has a single data root --------
    src = {
        "fig5_im_lineage_rate.csv":
            RDATA_X / "immune_metadata_HV4_EBV_positive_rate_celltype3_sample_positive_rate.csv",
    }
    tc6 = Path("F:/EBV/EBV_xiaomi/TC_EBV/06_ebv_inflammatory_exhaustion/out")
    for name in ("EBV_rate_by_group.csv", "EBV_rate_fisher_by_group.csv",
                 "EBV_rate_with_clinical.csv", "EBV_DNA_vs_CD8_EBV_rate_spearman.csv",
                 "HLH_EBV_pos_State_distribution.csv", "HLH_pex_EBV_module_wilcox.csv",
                 "HLH_pex_EBVpos_vs_neg_DE_all.csv", "HLH_pex_key_genes_EBVpos_vs_neg.csv"):
        src[name] = tc6 / name
    fig05 = Path("F:/EBV/EBV_xiaomi/EBV/figure_noIMM_regen/out/Fig05")
    src["cd8_ebv_by_celltype.csv"] = fig05 / "panel_k_cd8_ebv_by_celltype_noIMM.csv"
    src["cd8_ebv_by_state.csv"] = fig05 / "panel_l_cd8_ebv_by_state_noIMM.csv"

    for name, path in src.items():
        pd.read_csv(path).to_csv(OUT / name, index=False)
        print("copied", name)


if __name__ == "__main__":
    main()
