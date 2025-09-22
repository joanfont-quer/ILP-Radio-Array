from Partitioner.multipart_heuristic import wasserstein
from graph_loader import generate_graph
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import bisect

seeds = [137, 58291, 9021, 47717, 26539]

# load Gurobi data
df_gurobi = pd.read_parquet("/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_combined.parquet", engine="fastparquet")
df_gurobi = df_gurobi[df_gurobi["subarray_number"] == 2]
gurobi_solutions = dict(np.load("/share/nas2_3/jfont/ILP-Radio-Array/solution_files/solutions_combined.npz", allow_pickle=True))


# load heuristic data
df_heuristic = pd.read_parquet("/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_heuristic_1.parquet", engine="fastparquet")
solutions_heuristic = dict(np.load("/share/nas2_3/jfont/ILP-Radio-Array/solution_files/solutions_heuristic_1.npz", allow_pickle=True))


def wasserstein_true(x, y, p=1):
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = min(len(x_sorted), len(y_sorted))
    if p == 1:
        return stats.wasserstein_distance(x_sorted, y_sorted)
    else:
        return (np.mean(np.abs(x_sorted[:n] - y_sorted[:n]) ** p)) ** (1/p)
    

def partition_size_metrics(partition_weights):
    sizes = {pid: len(w) for pid, w in partition_weights.items()}
    ranges = {pid: (np.min(w), np.max(w)) if len(w) > 0 else (0, 0) for pid, w in partition_weights.items()}
    means = {pid: np.mean(w) if len(w) > 0 else 0 for pid, w in partition_weights.items()}
    return sizes, ranges, means
    

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


def compare_per_seed(node_num, subarray_number, seed):
    key = f"node{node_num}_sub{subarray_number}_seed{seed}"

    g_metadata = df_gurobi[df_gurobi["solution_key"] == key]
    g_solution = gurobi_solutions[key].item()

    h_metadata = df_heuristic[df_heuristic["solution_key"] == key]
    h_solution = solutions_heuristic[key]

    graph, _ = generate_graph(node_num, seed=seed)

    g_part_weight = create_partition_weights(graph, g_solution)
    h_part_weight = create_partition_weights(graph, h_solution)

    g_ks_stat, g_ks_p = stats.ks_2samp(g_part_weight[0], g_part_weight[1])
    h_ks_stat, h_ks_p = stats.ks_2samp(h_part_weight[0], h_part_weight[1])

    if len(g_part_weight[0]) > 1 and len(g_part_weight[1]) > 1:
        g_ad_result = stats.anderson_ksamp([g_part_weight[0], g_part_weight[1]])
        g_ad_stat = g_ad_result.statistic
        g_ad_p = g_ad_result.significance_level
    else:
        g_ad_stat, g_ad_p = np.nan, np.nan
        print(g_part_weight[0])
        print(g_part_weight[1])

    if len(h_part_weight[0]) > 1 and len(h_part_weight[1]) > 1:
        h_ad_result = stats.anderson_ksamp([h_part_weight[0], h_part_weight[1]])
        h_ad_stat = h_ad_result.statistic
        h_ad_p = h_ad_result.significance_level
    else:
        h_ad_stat, h_ad_p = np.nan, np.nan
        print(h_part_weight[0])
        print(h_part_weight[1])

    g_pad_l1 = wasserstein(g_part_weight, p=1)
    h_pad_l1 = h_metadata["cost"].iloc[0]

    g_pad_l2 = wasserstein(g_part_weight, p=2)
    h_pad_l2 = wasserstein(h_part_weight, p=2)

    g_true_l1 = wasserstein_true(g_part_weight[0], g_part_weight[1], p=1)
    h_true_l1 = wasserstein_true(h_part_weight[0], h_part_weight[1], p=1)

    g_true_l2 = wasserstein_true(g_part_weight[0], g_part_weight[1], p=2)
    h_true_l2 = wasserstein_true(h_part_weight[0], h_part_weight[1], p=2)

    g_sizes, g_ranges, g_means = partition_size_metrics(g_part_weight)
    h_sizes, h_ranges, h_means = partition_size_metrics(h_part_weight)
    g_size_diff = abs(g_sizes[0] - g_sizes[1])
    h_size_diff = abs(h_sizes[0] - h_sizes[1])
    g_range_diff = abs((g_ranges[0][1] - g_ranges[0][0]) - (g_ranges[1][1] - g_ranges[1][0]))
    h_range_diff = abs((h_ranges[0][1] - h_ranges[0][0]) - (h_ranges[1][1] - h_ranges[1][0]))

    results = pd.DataFrame({
        "Metric": [
            "KS Statistic",
            "KS p-value",
            "Anderson–Darling Statistic",
            "Anderson–Darling p-value",
            "Padded Wasserstein L1",
            "Padded Wasserstein L2",
            "True Wasserstein L1",
            "True Wasserstein L2",
            "Partition size difference (edges)",
            "Partition range difference (max-min)",
        ],
        "Gurobi": [
            g_ks_stat,
            g_ks_p,
            g_ad_stat,
            g_ad_p,
            g_pad_l1,
            g_pad_l2,
            g_true_l1,
            g_true_l2,
            g_size_diff,
            g_range_diff
        ],
        "Heuristic": [
            h_ks_stat,
            h_ks_p,
            h_ad_stat,
            h_ad_p,
            h_pad_l1,
            h_pad_l2,
            h_true_l1,
            h_true_l2,
            h_size_diff,
            h_range_diff
        ]
    })
    return results


def main(node_num, subarray_number):

    g_p_val = []
    h_p_val = []
    for n in range(np.min(df_gurobi["node_num"]), np.max(df_gurobi["node_num"] + 1)):
        g_p_vals = []
        h_p_vals = []
        for seed in seeds:
            key = f"node{n}_sub{subarray_number}_seed{seed}"

            g_solution = gurobi_solutions[key].item()

            h_solution = solutions_heuristic[key]

            graph, _ = generate_graph(n, seed=seed)

            g_part_weight = create_partition_weights(graph, g_solution)
            h_part_weight = create_partition_weights(graph, h_solution)

            _, g_ks_p = stats.ks_2samp(g_part_weight[0], g_part_weight[1])
            _, h_ks_p = stats.ks_2samp(h_part_weight[0], h_part_weight[1])

            g_p_vals.append(g_ks_p)
            h_p_vals.append(h_ks_p)
        g_p_val.append(np.mean(g_p_vals))
        h_p_val.append(np.mean(h_p_vals))
    
    plt.figure(figsize=(6,6))
    plt.plot(g_p_val, h_p_val, "o", label="KS p-values", linestyle='None')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0,1], [0,1], "k--", label="Equal p-values")
    for n, gx, hx in zip(range(np.min(df_gurobi["node_num"]), np.max(df_gurobi["node_num"])), g_p_val, h_p_val):
        plt.text(gx, hx, str(n), fontsize=8)  # annotate node numbers

    plt.xlabel("Gurobi KS p-value")
    plt.ylabel("Heuristic KS p-value")
    plt.legend()
    plt.savefig("g_vs_h_pval.png")
    plt.close()


    metric_results = []
    for n in range(np.min(df_gurobi["node_num"])+1, np.max(df_gurobi["node_num"])+1):
        per_node = []
        for seed in seeds:
            metrics = compare_per_seed(n, 2, seed)
            per_node.append(metrics)
        combined = pd.concat(per_node).groupby("Metric").mean(numeric_only=True).reset_index()
        combined["node_num"] = n
        metric_results.append(combined)
    
    metrics_all = pd.concat(metric_results)

    unique_metrics = metrics_all["Metric"].unique()
    n_metrics = len(unique_metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for idx, metric in enumerate(unique_metrics):
        ax = axes[idx // ncols, idx % ncols]

        subset = metrics_all[metrics_all["Metric"] == metric]

        ax.plot(subset["Gurobi"], subset["Heuristic"], "o", label=metric)
        ax.plot([subset["Gurobi"].min(), subset["Gurobi"].max()],
                [subset["Gurobi"].min(), subset["Gurobi"].max()],
                "k--", label="y=x")

        for _, row in subset.iterrows():
            ax.text(row["Gurobi"], row["Heuristic"], str(int(row["node_num"])), fontsize=7)

        ax.set_xlabel("Gurobi")
        ax.set_ylabel("Heuristic")
        ax.set_title(metric)
        ax.legend()

        xmin = min(subset["Gurobi"].min(), subset["Heuristic"].min())
        xmax = max(subset["Gurobi"].max(), subset["Heuristic"].max())
        ax.set_xlim(xmin*1.1, xmax*1.1)
        ax.set_ylim(xmin*1.1, xmax*1.1)

    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["True Wasserstein L1", "KS p-value"]):
        ax = axes[idx]

        subset = metrics_all[metrics_all["Metric"] == metric]

        xmin = min(subset["Gurobi"].min(), subset["Heuristic"].min())
        xmax = max(subset["Gurobi"].max(), subset["Heuristic"].max())
        ax.set_xlim(xmin*1.2, xmax*1.2)
        ax.set_ylim(xmin*1.2, xmax*1.2)

        node_nums = subset["node_num"].values

        norm = mcolors.Normalize(vmin=node_nums.min(), vmax=node_nums.max())
        colors = cm.viridis(norm(node_nums))

        ax.scatter(subset["Gurobi"], subset["Heuristic"], c=colors, s=50)
        ax.plot([xmin*1.2, xmax*1.2],
                [xmin*1.2, xmax*1.2],
                "k--", label="y=x")

        for _, row in subset.iterrows():
            ax.text(row["Gurobi"], row["Heuristic"], str(int(row["node_num"])), fontsize=7)

        ax.set_xlabel("Gurobi")
        ax.set_ylabel("Heuristic")
        ax.legend()

        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
        cbar.set_label("Node number")
    
    plt.tight_layout()
    plt.savefig("metrics_report.png")
    plt.close()


if __name__ == '__main__':
    main(45, 2)
