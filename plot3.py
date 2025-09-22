from graph_loader import generate_graph
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import bisect


def create_partition_weights(graph, solution):
    partition_weights = {}
    for part_id in [0, 1]:
        weights = []
        for u, v in graph.edges():
            if solution[u] == part_id and solution[v] == part_id:
                w = graph[u][v]["weight"]
                bisect.insort(weights, w)
        partition_weights[part_id] = [x for x in weights]
    return partition_weights


# load data
df = pd.read_parquet("/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_bins.parquet", engine="fastparquet")
df_node = df[df["node_num"] == 40]
df_sub2 = df_node[df_node["subarray_number"] == 2]
df_bin68 = df_sub2[df_sub2["bin_number"] <= 68]

gurobi_solutions = dict(np.load("/share/nas2_3/jfont/ILP-Radio-Array/solution_files/solutions_bins.npz", allow_pickle=True))


# compute the mean and std of the optimisation time per antenna number
stats = df_bin68.groupby("bin_number")["optimisation_time"].agg(["median", "mean", "std"]).reset_index()
stats["mean"] = stats["mean"] / 3600
stats["std"] = stats["std"] / 3600

ks_p_vals = []
ks_stds = []
for bin in range(np.min(df["bin_number"]), 69):
    ks_p_bin = []
    for seed in [137, 58291, 9021, 47717, 26539]:
        graph, _ = generate_graph(40, seed=seed)

        solution = gurobi_solutions[f"node40_sub2_seed{seed}_bin{bin}"].item()
        partition_weights = create_partition_weights(graph, solution)

        _, ks_p = st.ks_2samp(partition_weights[0], partition_weights[1])
        ks_p_bin.append(ks_p)
    ks_p_vals.append(np.mean(ks_p_bin))
    ks_stds.append(np.std(ks_p_bin))


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

axes[0].errorbar(
    stats["bin_number"], stats["mean"], yerr=stats["std"], fmt="o-", ecolor="gray",
    capsize=4, markersize=5, label="Optimisation Time"
)
axes[0].set_xlabel("Number of Bins")
axes[0].set_ylabel("Optimisation Time (hours)")
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].errorbar(
    stats["bin_number"], ks_p_vals, yerr=ks_stds, fmt="s-", ecolor="gray",
    capsize=4, markersize=5, label="KS p-value"
)
axes[1].set_xlabel("Number of Bins")
axes[1].set_ylabel("Mean KS p-value")
axes[1].set_ylim(0, 1)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("plots/opt_and_ks_vs_bin_num_node40.png")
plt.close()
