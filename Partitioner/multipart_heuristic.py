import random
import bisect
import graph_loader as gl
import visualisation as vis
import numpy as np


def wasserstein(subarray_weights, p):
    total_was = 0
    subarrays = list(subarray_weights.keys())
    for i in range(len(subarrays)):
        for j in range(i+1, len(subarrays)):
            weights_i = np.array(subarray_weights[subarrays[i]])
            weights_j = np.array(subarray_weights[subarrays[j]])

            if len(weights_i) > len(weights_j):
                weights_j = np.pad(weights_j, (0, len(weights_i) - len(weights_j)), 'constant')
            elif len(weights_j) >len(weights_i):
                weights_i = np.pad(weights_i, (0, len(weights_j) - len(weights_i)), 'constant')
            total_was += np.sum(np.abs(weights_i - weights_j) ** p) ** (1/p)
    return total_was


def update_weight_list(graph, y, subarray_weights, node):
    for neighbor in graph.neighbors(node):
        if y[neighbor] == y[node]:
            bisect.insort(subarray_weights[y[node]], graph[node][neighbor]['weight'])
    return subarray_weights


def assign_loop(graph, y, unassigned_node_list, subarray_weights, subarray_number, p):
    best_dist = float('inf')
    best_assignment = None
    new_subarray_weights = None

    for node in unassigned_node_list:
        for subarray in range(subarray_number):
            y[node] = subarray
            temp_weights = {k: v.copy() for k, v in subarray_weights.items()}
            temp_weights = update_weight_list(graph, y, temp_weights, node)
            current_dist = wasserstein(temp_weights, p)
            if current_dist < best_dist:
                best_dist = current_dist
                best_assignment = (node, subarray)
                new_subarray_weights = temp_weights
            y[node] = -1

    node, subarray = best_assignment
    y[node] = subarray
    unassigned_node_list.remove(node)
    subarray_weights = new_subarray_weights
    return y, unassigned_node_list, subarray_weights


def solver(graph, subarray_number, p=1):
    unassigned_node_list = list(graph.nodes())

    y = [-1] * len(graph.nodes())

    for subarray in range(subarray_number):
        assigned_node = random.choice(unassigned_node_list)
        y[assigned_node] = subarray
        unassigned_node_list.remove(assigned_node)

    subarray_weights = {g: [] for g in range(subarray_number)}

    while len(unassigned_node_list) > 0:
        y, unassigned_node_list, subarray_weights = assign_loop(graph, y, unassigned_node_list, subarray_weights, subarray_number, p)

    cost = wasserstein(subarray_weights, p)
    return y, cost
