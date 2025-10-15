import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import bisect
import re
from Partitioner.objectives import wasserstein


class PartitionBalanceBenchmark:
    def __init__(self, config1, config2, graph_func, output_dir, x_param="node_num"):
        self.config1 = config1
        self.config2 = config2
        self.metric_results = None
        self.output_dir = Path(output_dir)
        self.graph_func = graph_func
        self.x_param = x_param
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _create_partition_weights(graph, solution):
        partition_weights = {}
        partition_ids = range(max(solution.values()) + 1)
        for part_id in partition_ids:
            weights = []
            for u, v in graph.edges():
                if solution[u] == part_id and solution[v] == part_id:
                    bisect.insort(weights, graph[u][v]["weight"])
            partition_weights[part_id] = weights
        return partition_weights

    @staticmethod
    def _partition_size_metrics(partition_weights):
        sizes = {pid: len(w) for pid, w in partition_weights.items()}
        ranges = {pid: (np.min(w), np.max(w)) if len(w) > 0 else (0, 0) for pid, w in partition_weights.items()}
        means = {pid: np.mean(w) if len(w) > 0 else 0 for pid, w in partition_weights.items()}
        return sizes, ranges, means

    def _get_all_metrics(self, partition_weights):
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

    def compare(self):
        graph, *rest = self.graph_func()

        config1_weight = self._create_partition_weights(graph, self.config1.get_ska_solution())
        config2_weight = self._create_partition_weights(graph, self.config2.get_ska_solution())

        config1_metrics = self._get_all_metrics(config1_weight)
        config2_metrics = self._get_all_metrics(config2_weight)

        df = pd.DataFrame({
            "Metric": list(config1_metrics.keys()),
            f"{self.config1.NAME}": list(config1_metrics.values()),
            f"{self.config2.NAME}": list(config2_metrics.values())
        })
        return df

    def _compare_per_seed(self, node_num, subarray_num, seed):
        key = f"node{node_num}_sub{subarray_num}_seed{seed}"
        sol1 = self.config1.solutions()[key]
        sol2 = self.config2.solutions()[key]

        graph, *rest = self.graph_func(node_num, seed=seed)

        w1 = self._create_partition_weights(graph, sol1)
        w2 = self._create_partition_weights(graph, sol2)

        metrics1 = self._get_all_metrics(w1)
        metrics2 = self._get_all_metrics(w2)

        return metrics1, metrics2

    @staticmethod
    def _extract_seed(key):
        match = re.search(r"seed(\d+)", key)
        return int(match.group(1)) if match else None

    def _get_common_seeds(self, df1, df2):
        seeds1 = set(df1["solution_key"].apply(self._extract_seed))
        seeds2 = set(df2["solution_key"].apply(self._extract_seed))
        return sorted(seeds1.intersection(seeds2))

    def run(self, subarray_num):
        df1 = self.config1.metadata()
        df2 = self.config2.metadata()

        seeds = self._get_common_seeds(df1, df2)

        x_values = sorted(set(df1[self.x_param]).intersection(df2[self.x_param]))

        metric_results = []
        for x in x_values:
            for seed in seeds:
                per_seed1, per_seed2 = self._compare_per_seed(x, subarray_num, seed)
                for config_name, metrics in [
                (self.config1.NAME, per_seed1),
                (self.config2.NAME, per_seed2)
            ]:
                    row = {self.x_param: x, "seed": seed, "config": config_name}
                    row.update(metrics)
                    metric_results.append(row)
        df_metrics = pd.DataFrame(metric_results)
        self.metric_results = df_metrics
        return df_metrics

    def plot_metric_comparison(self, metric_name):
        df = self.metric_results

        df_mean = df.groupby([self.x_param, "config"], as_index=False)[metric_name].mean()

        df_wide = df_mean.pivot(
            index=self.x_param, columns="config", values=metric_name
        ).reset_index()

        x = df_wide[self.config1.NAME]
        y = df_wide[self.config2.NAME]

        x_mean, x_std = x.mean(), x.std()
        y_mean, y_std = y.mean(), y.std()

        x_min, x_max = x_mean - 3 * x_std, x_mean + 3 * x_std
        y_min, y_max = y_mean - 3 * y_std, y_mean + 3 * y_std

        node_nums = df_wide[self.x_param]

        plt.figure(figsize=(6, 6))
        norm = plt.Normalize(node_nums.min(), node_nums.max())

        sc = plt.scatter(x, y, c=node_nums, cmap="viridis", norm=norm, s=50)
        plt.plot([x.min(), x.max()], [x.min(), x.max()], "k--", label="y=x")

        axis_range = min(x_max, y_max)

        plt.xlim(0, axis_range)
        plt.ylim(0, axis_range)

        for i, n in enumerate(node_nums):
            plt.text(x.iloc[i], y.iloc[i], str(n), fontsize=7)

        plt.xlabel(f"{self.config1.NAME} {metric_name}")
        plt.ylabel(f"{self.config2.NAME} {metric_name}")
        plt.title(f"Comparison of {metric_name}")

        cbar = plt.colorbar(sc)
        cbar.set_label(self.x_param)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(self.output_dir / f"{self.config1.NAME}_vs_{self.config2.NAME}_{metric_name}.png", dpi=300)
        plt.close()
