import os
import numpy as np
import pandas as pd
from scipy import stats
import bisect
from ska_ost_array_config import get_subarray_template
import graph_loader as gl
from Partitioner.multipart_heuristic import wasserstein
import matplotlib.pyplot as plt
import configs.ska_2_subarray as ska


class PartitionBalanceBenchmark:
    def __init__(self, config, x_param, output_dir):
        self.config = config
        self.x_param = x_param
        self.output_dir = output_dir
        self.graph_func = gl.ska_mid_full_graph


    def _create_partition_weights(self, graph, solution):
        partition_weights = {}
        for part_id in [0, 1]:
            weights = []
            for u, v in graph.edges():
                if solution[u] == part_id and solution[v] == part_id:
                    bisect.insort(weights, graph[u][v]["weight"])
            partition_weights[part_id] = weights
        return partition_weights

    def _partition_size_metrics(self, partition_weights):
        sizes = {pid: len(w) for pid, w in partition_weights.items()}
        ranges = {pid: (np.min(w), np.max(w)) if len(w) > 0 else (0, 0) for pid, w in partition_weights.items()}
        means = {pid: np.mean(w) if len(w) > 0 else 0 for pid, w in partition_weights.items()}
        return sizes, ranges, means

    def compare(self, solution):
        graph, _, _ = self.graph_func()

        part_weight = self._create_partition_weights(graph, solution)
        ska_part_weight = self._create_partition_weights(graph, ska.get_ska_solution())

        def get_all_metrics(partition_weights):
            ks_stat, ks_p = stats.ks_2samp(partition_weights[0], partition_weights[1])

            if len(partition_weights[0]) > 1 and len(partition_weights[1]) > 1:
                ad_result = stats.anderson_ksamp([partition_weights[0], partition_weights[1]])
                ad_stat = ad_result.statistic
                ad_p = ad_result.significance_level
            else:
                ad_stat, ad_p = np.nan, np.nan

            l1 = wasserstein(partition_weights, {"p": 1, "alpha": 1})
            l2 = wasserstein(partition_weights, {"p": 2, "alpha": 1})

            sizes, ranges, means = self._partition_size_metrics(partition_weights)
            size_diff = abs(sizes[0] - sizes[1])
            range_diff = abs((ranges[0][1] - ranges[0][0]) - (ranges[1][1] - ranges[1][0]))

            return dict(ks_stat=ks_stat, ks_p=ks_p, ad_stat=ad_stat, ad_p=ad_p,
                        l1=l1, l2=l2, size_diff=size_diff, range_diff=range_diff)

        part_metrics = get_all_metrics(part_weight)
        ska_metrics = get_all_metrics(ska_part_weight)

        df = pd.DataFrame({
            "Metric": list(part_metrics.keys()),
            "Current": list(ska_metrics.values()),
            "Partition": list(part_metrics.values())
        })
        return df

