import numpy as np
from scipy.stats._stats_py import _cdf_distance


def wasserstein(partition_weights, params):
    """
        Computes the normalised pairwise Wasserstein distance of order 'p' between all pairs of partitions.

        Args:
            partition_weights (dict): Dictionary with ordered edge weights for each partition.
            params (dict): Dictionary containing objective parameters:
                - 'p' (float): Order of Wasserstein distance.
                - 'alpha' (float): Parameter controlling penalty on variance of the partition sizes.

        Returns:
            float: Total sum of pairwise Wasserstein distances across all partitions, plus the penalty term on
                   partition imbalances.
        """
    p = params.get('p', 1.0)
    alpha = params.get('alpha', 1.0)

    all_weights = np.concatenate([np.asarray(w) for w in partition_weights.values() if len(w) > 0])
    max_w = np.max(all_weights)

    total = 0.0
    partitions = list(partition_weights.keys())
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            w_i = np.array(partition_weights[partitions[i]]) / max_w
            w_j = np.array(partition_weights[partitions[j]]) / max_w

            if (len(w_i) == 0) or (len(w_j) == 0):
                total += 1e6
            else:
                wasserstein_p = _cdf_distance(p, w_i, w_j)

                total += wasserstein_p
    sizes = np.array([len(partition_weights[part]) for part in partitions])

    node_nums = (1 + np.sqrt(sizes*8 + 1)) / 2
    return total + alpha * np.var(node_nums) / (np.mean(node_nums) ** 2)
