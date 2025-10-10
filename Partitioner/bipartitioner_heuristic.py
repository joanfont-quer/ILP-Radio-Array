import random
import graph_loader as gl
import visualisation as vis
import numpy as np


def wasserstein(y, graph):
    weights_0 = []
    weights_1 = []

    for u, v, data in graph.edges(data=True):
        weight = data['weight']

        if y[u] == 0 and y[v] == 0:
            weights_0.append(weight)
        elif y[u] == 1 and y[v] == 1:
            weights_1.append(weight)

    weights_0 = np.array(weights_0)
    weights_1 = np.array(weights_1)

    weights_0.sort()
    weights_1.sort()

    wasserstein_distance = np.sum(np.abs(weights_0 - weights_1))
    return wasserstein_distance


def assign_loop(y, graph, unassigned_node_list):
    best_dist = float('inf')
    best_assignment = [0, 0]
    for i in unassigned_node_list:
        for j in unassigned_node_list:
            if i != j:
                y[i] = 0
                y[j] = 1
                was_dist = wasserstein(y, graph)
                if was_dist < best_dist:
                    best_dist = was_dist
                    best_assignment = (i, j)

                y[i] = -1
                y[j] = -1

    i, j = best_assignment
    y[i] = 0
    y[j] = 1
    unassigned_node_list.remove(i)
    unassigned_node_list.remove(j)
    return y, unassigned_node_list


def solver(graph):
    unassigned_node_list = list(graph.nodes())
    # y 0 if in subarray 0, 1 if in subarray 1
    y = [-1] * len(graph.nodes())
    # assign random node to subarray 0
    assigned_node = random.choice(unassigned_node_list)
    y[assigned_node] = 0
    unassigned_node_list.remove(assigned_node)

    # assign random node to subarray 1
    assigned_node = random.choice(unassigned_node_list)
    y[assigned_node] = 1
    unassigned_node_list.remove(assigned_node)

    while len(unassigned_node_list) > 0:
        y, unassigned_node_list = assign_loop(y, graph, unassigned_node_list)

    return y
