from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from Partitioner.utils import *


def setup(graph, subarray_number):
    n = len(graph.nodes)
    problem = LpProblem("RadioArrayDivider", LpMinimize)

    y = {
        (i, s): LpVariable(f"y_{i}_{s}", 0, 1, cat="Binary")
        for i in range(n)
        for s in range(subarray_number)
    }

    # antenna 'i' can only be assigned to one subarray
    for i in range(n):
        problem += lpSum(y[i, s] for s in range(subarray_number)) == 1

    z = {}
    for s in range(subarray_number):
        for i in range(n):
            for j in range(i + 1, n):
                z[(i, j, s)] = LpVariable(f"z_{i}_{j}_{s}", 0, 1, cat="Binary")
                problem += z[(i, j, s)] <= y[i, s]
                problem += z[(i, j, s)] <= y[j, s]
                problem += z[(i, j, s)] >= y[i, s] + y[j, s] - 1

    return problem, y, z, n


def sum_edges(graph, z, subarray_number):
    active_edges = [(i, j) for i, j, d in graph.edges(data=True) if d['weight'] > 0]
    sums = []
    for s in range(subarray_number):
        sum_s = lpSum(z[(i, j, s)] for i, j in active_edges)
        sums.append(sum_s)

    return sums


def solve_bins(graph, bin_number, subarray_number):
    problem, y, z, n = setup(graph, subarray_number)

    bins = bin_maker(graph, bin_number)

    diff_vars = []
    for b in bins:
        masked_graph = graph_masker(graph, b[0], b[1])
        bin_sums = sum_edges(masked_graph, z, subarray_number)

        sum_max = LpVariable(f"sum_max_{b[0]}_{b[1]}", 0)
        sum_min = LpVariable(f"sum_min_{b[0]}_{b[1]}", 0)

        for s in range(subarray_number):
            problem += sum_max >= bin_sums[s]
            problem += sum_min <= bin_sums[s]

        diff = LpVariable(f"diff_{b[0]}_{b[1]}", lowBound=0)
        problem += diff >= sum_max - sum_min

        diff_vars.append(diff)

    problem += lpSum(diff_vars)
    problem.solve(PULP_CBC_CMD(msg=False, threads=8))

    subarrays = {}
    for i in graph.nodes():
        for s in range(subarray_number):
            if y[i, s].varValue == 1:
                subarrays[i] = s

    cost = problem.objective.value()

    return subarrays, cost
