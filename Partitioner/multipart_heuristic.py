import random
import bisect
import numpy as np


def wasserstein(partition_weights, p):
    """
    Computes the pairwise Wasserstein distance of order 'p' between all pairs of partitions.

    Args:
        partition_weights (dict): Dictionary with ordered edge weights for each partition.
        p (float): Order of Wasserstein distance.

    Returns:
        float: Total sum of pairwise Wasserstein distances across all partitions.
    """
    total = 0.0
    partitions = list(partition_weights.keys())
    for i in range(len(partitions)):
        for j in range(i+1, len(partitions)):
            w_i = np.array(partition_weights[partitions[i]])
            w_j = np.array(partition_weights[partitions[j]])

            if len(w_i) == 0 and len(w_j) == 0:
                continue
            elif len(w_i) > len(w_j):
                w_j = np.pad(w_j, (0, len(w_i) - len(w_j)), 'constant')
            elif len(w_j) > len(w_i):
                w_i = np.pad(w_i, (0, len(w_j) - len(w_i)), 'constant')

            total += np.mean(np.abs(w_i - w_j) ** p) ** (1/p)
    return total


def update_partition_weights(graph, y, partition_weights, node):
    """
    Updates the weight list of the partition containing 'node' by including the weights of edges connecting 'node' to
    its neighbours in the same partition.

    Args:
        graph (networkx.Graph): The input graph.
        y (list): Partition assignment list for each node.
        partition_weights (dict): Dictionary with ordered edge weights for each partition.
        node (int): Node to update the weight dictionary for.

    Returns:
        dict: Updated partition_weights with new weights from `node`.
    """
    for neighbor in graph.neighbors(node):
        if y[neighbor] == y[node]:
            bisect.insort(partition_weights[y[node]], graph[node][neighbor]['weight'])
    return partition_weights


def assign_node(graph, y, unassigned, partition_weights, partition_number, p):
    """
    Assigns one unassigned node to a partition while minimising the overall Wasserstein cost.

    Args:
        graph (networkx.Graph): The input graph.
        y (list): Partition assignment list for each node.
        unassigned (list): List of unassigned node indices.
        partition_weights (dict): Dictionary with ordered edge weights for each partition.
        partition_number (int): Number of partitions.
        p (float): Order of Wasserstein distance.

    Returns:
        Tuple: Updated y, unassigned, and partition_weights after assignment.
    """
    best_dist = float('inf')
    best_assignment = None, None
    new_partition_weights = None

    for node in unassigned:
        for partition in range(partition_number):
            y_temp = y.copy()
            y_temp[node] = partition

            temp_weights = {k: v.copy() for k, v in partition_weights.items()}
            temp_weights = update_partition_weights(graph, y_temp, temp_weights, node)

            current_dist = wasserstein(temp_weights, p)
            if current_dist < best_dist:
                best_dist = current_dist
                best_assignment = node, partition
                new_partition_weights = temp_weights

    node, partition = best_assignment
    y[node] = partition
    unassigned.remove(node)
    partition_weights = new_partition_weights
    return y, unassigned, partition_weights


def solver(graph, partition_num, p=1.0, seed=42):
    """
    Assigns all nodes in a graph to 'partition_num' partitions such that the total Wasserstein distance of order 'p'
    between partition edge weight distributions is approximately minimised.

    Args:
        graph (networkx.Graph): The input graph.
        partition_num (int): Number of partitions.
        p (float): Wasserstein order.
        seed (int): Random seed.

    Returns:
        Tuple: Final partition assignments (list) and total Wasserstein cost (float).
    """
    random.seed(seed)
    unassigned = list(graph.nodes())

    y = {node: -1 for node in graph.nodes()}

    for partition in range(partition_num):
        node = random.choice(unassigned)
        y[node] = partition
        unassigned.remove(node)

    partition_weights = {g: [] for g in range(partition_num)}

    while len(unassigned) > 0:
        y, unassigned, partition_weights = assign_node(graph, y, unassigned, partition_weights, partition_num, p)

    cost = wasserstein(partition_weights, p)
    return y, cost
