"""Stage the Figure 8 tables that already exist as CSV upstream.

The monocyte embedding, markers and HVG panels come from export_fig8_data.R;
this only pulls in the CellChat interaction table and the ferritin join so the
plotting script reads everything from one directory.
"""
from pathlib import Path
import pandas as pd

SRC_CHAT = Path(r"F:\EBV\EBV_xiaomi\EBV\Rdata\cellchat_MC\MC_to_CD8_all_newgroup6.csv")
SRC_FERR = Path(r"F:\EBV\EBV_xiaomi\EBV\figure_noIMM_regen\out\Fig08"
                r"\panel_k_ferritin_join_noIMM.csv")
OUT = Path(r"F:\EBV\EBV_xiaomi\Figures_Nature_Python\_data\fig8")

GROUPS = ["HC", "IM", "MH_CD4", "HLH"]
SOURCE_MC = "MC_MC_NLRP3_HK2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    chat = pd.read_csv(SRC_CHAT)
    chat = chat[(chat["source"] == SOURCE_MC) & chat["Newgroup6"].isin(GROUPS)].copy()
    chat["CD8_subtype"] = chat["target"].str.replace(r"^CD8_CD8T_", "", regex=True)
    cols = ["Newgroup6", "CD8_subtype", "ligand", "receptor", "interaction_name",
            "interaction_name_2", "pathway_name", "prob", "pval"]
    chat[cols].to_csv(OUT / "fig8_cellchat.csv", index=False)
    print(f"cellchat: {len(chat)} MC_NLRP3_HK2 -> CD8 interactions, "
          f"{chat['CD8_subtype'].nunique()} CD8 subtypes, "
          f"{chat['interaction_name'].nunique()} L-R pairs")

    ferr = pd.read_csv(SRC_FERR)
    ferr.to_csv(OUT / "fig8_ferritin.csv", index=False)
    print(f"ferritin: {len(ferr)} participants "
          f"({ferr['Newgroup6'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
