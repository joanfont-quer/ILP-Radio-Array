import numpy as np
import pandas as pd
from scipy import stats
import bisect
from ska_ost_array_config import get_subarray_template
import graph_loader as gl
from Partitioner.multipart_heuristic import wasserstein
    

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


def compare(h_solution, current_ska):
    graph, _, _ = gl.ska_mid_full_graph()

    h_part_weight = create_partition_weights(graph, h_solution)
    ska_part_weight = create_partition_weights(graph, current_ska)

    h_ks_stat, h_ks_p = stats.ks_2samp(h_part_weight[0], h_part_weight[1])
    ska_ks_stat, ska_ks_p = stats.ks_2samp(ska_part_weight[0], ska_part_weight[1])

    if len(h_part_weight[0]) > 1 and len(h_part_weight[1]) > 1:
        h_ad_result = stats.anderson_ksamp([h_part_weight[0], h_part_weight[1]])
        h_ad_stat = h_ad_result.statistic
        h_ad_p = h_ad_result.significance_level
    else:
        h_ad_stat, h_ad_p = np.nan, np.nan

    if len(ska_part_weight[0]) > 1 and len(ska_part_weight[1]) > 1:
        ska_ad_result = stats.anderson_ksamp([ska_part_weight[0], ska_part_weight[1]])
        ska_ad_stat = ska_ad_result.statistic
        ska_ad_p = ska_ad_result.significance_level
    else:
        ska_ad_stat, ska_ad_p = np.nan, np.nan

    h_pad_l1 = wasserstein(h_part_weight, {"p": 1, "alpha": 1})
    ska_pad_l1 = wasserstein(ska_part_weight, {"p": 1, "alpha": 1})

    h_pad_l2 = wasserstein(h_part_weight, {"p": 2, "alpha": 1})
    ska_pad_l2 = wasserstein(ska_part_weight, {"p": 2, "alpha": 1})

    h_sizes, h_ranges, h_means = partition_size_metrics(h_part_weight)
    ska_sizes, ska_ranges, ska_means = partition_size_metrics(ska_part_weight)
    h_size_diff = abs(h_sizes[0] - h_sizes[1])
    ska_size_diff = abs(ska_sizes[0] - ska_sizes[1])
    h_range_diff = abs((h_ranges[0][1] - h_ranges[0][0]) - (h_ranges[1][1] - h_ranges[1][0]))
    ska_range_diff = abs((ska_ranges[0][1] - ska_ranges[0][0]) - (ska_ranges[1][1] - ska_ranges[1][0]))

    results = pd.DataFrame({
        "Metric": [
            "KS Statistic",
            "KS p-value",
            "Anderson–Darling Statistic",
            "Anderson–Darling p-value",
            "Wasserstein L1",
            "Wasserstein L2",
            "Partition size difference (edges)",
            "Partition range difference (max-min)",
        ],
        "Current": [
            ska_ks_stat,
            ska_ks_p,
            ska_ad_stat,
            ska_ad_p,
            ska_pad_l1,
            ska_pad_l2,
            ska_size_diff,
            ska_range_diff
        ],
        "Heuristic": [
            h_ks_stat,
            h_ks_p,
            h_ad_stat,
            h_ad_p,
            h_pad_l1,
            h_pad_l2,
            h_size_diff,
            h_range_diff
        ]
    })
    return results


def main():
    solutions = dict(np.load("/ILP-Radio-Array/solutions_heuristic_ska_full.npz", allow_pickle=True))
    print(solutions.items())

    key = "node197_sub2_seed137"

    h_solution = solutions[key].item()

    split_1 = get_subarray_template("MID_SPLIT2_1_AA4")
    split_1_names = split_1.array_config.names.data
    split_1_names = [str(name) for name in split_1_names]

    split_2 = get_subarray_template("MID_SPLIT2_2_AA4")
    split_2_names = split_2.array_config.names.data
    split_2_names = [str(name) for name in split_2_names]

    ska_solution = {}
    for name in split_1_names:
        ska_solution[name] = 0
    for name in split_2_names:
        ska_solution[name] = 1

    results = compare(h_solution, ska_solution)

    print("=== Comparison Metrics ===")
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
