import pandas as pd
import numpy as np
from pathlib import Path

bin_key = True

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

files = ["/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_compute-0-2.local_42208.parquet", 
         "/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_compute-0-2.local_12744.parquet", 
         "/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_compute-0-2.local_12744_1.parquet",
         "/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_compute-0-2.local_12744_2.parquet"]

df_list = [pd.read_parquet(f, engine="fastparquet") for f in files]
df = pd.concat(df_list, ignore_index=True)

if bin_key == True:
    mask_missing_bin = ~df["solution_key"].str.contains(r"_bin\d+$")
    df.loc[mask_missing_bin, "solution_key"] = (
        df.loc[mask_missing_bin, "solution_key"] + "_bin" + df.loc[mask_missing_bin, "bin_number"].astype(str)
    )

best_df = df.loc[df.groupby("solution_key")["optimisation_time"].idxmin()]
best_df.to_parquet("metadata_files/metadata_bins.parquet", index=False)

solutions_all = {}
for metadata_path in files:
    suffix = metadata_path.split("metadata_")[-1].replace(".parquet", "")
    solution_file = Path(f"/share/nas2_3/jfont/ILP-Radio-Array/solution_files/solutions_{suffix}.npz")
    if solution_file.exists():
        sols_part = dict(np.load(solution_file, allow_pickle=True))
        solutions_all.update(sols_part)

best_keys = set(best_df["solution_key"])
solutions_filtered = {k: v for k, v in solutions_all.items() if k in best_keys}

np.savez_compressed("solution_files/solutions_bins.npz", **solutions_filtered)
