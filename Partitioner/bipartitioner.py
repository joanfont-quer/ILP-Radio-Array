from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from Partitioner.utils import *


def setup(graph):
    """
    Sets up used variables in the problem.
    Args:
        graph: networkx graph of all antennas

    Returns:
        problem: pulp problem
        y: antenna variable, 1 if antenna 'i' is in subarray 1, 0 if antenna 'i' in subarray 0
        z: auxiliary variable equal to y_i * y_j
        n: number of antennas
    """
    n = len(graph.nodes)
    problem = LpProblem("RadioArrayDivider", LpMinimize)

    y = {
        i: LpVariable(f"y_{i}", 0, 1, cat="Binary")
        for i in range(n)
    }

    z = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                z[(i,j)] = LpVariable(f"z_{i}_{j}", 0, 1, cat="Binary")
                problem += z[(i,j)] <= y[i]
                problem += z[(i,j)] <= y[j]
                problem += z[(i,j)] >= y[i] + y[j] - 1

    return problem, y, z, n


def sum_edges(graph, y, z, n):
    """
    Defines the sum of the weights of all the edges in subarray 0 and 1.
    Args:
        graph: networkx graph
        y: antenna variable, 1 if antenna 'i' is in subarray 1, 0 if antenna 'i' in subarray 0
        z: auxiliary variable equal to y_i * y_j
        n: number of antennas

    Returns:
        sum0: sum of the weights  of all edges in subarray 0
        sum1: sum of the weights  of all edges in subarray 1
    """
    sum0 = lpSum(graph[i][j]['weight'] * z[(i, j)]
                 for i in range(n)
                 for j in range(n)
                 if i != j)

    sum1 = lpSum(graph[i][j]['weight'] * (1 - y[i] - y[j] + z[(i, j)])
                 for i in range(n)
                 for j in range(n)
                 if i != j)
    return sum0, sum1


def solve_total(graph):
    """
    Minimises the difference in the sum of all baselines in subarrays 0 and 1.
    Args:
        graph: networkx graph of all antennas

    Returns:
        subarrays: directory where each antenna is assigned a value of either 0 or 1 depending on
                   which subarray it is in
        cost: difference between the sum of all baselines in subarray 0 and 1
    """
    problem, y, z, n = setup(graph)

    sum0, sum1 = sum_edges(graph, y, z, n)

    diff = LpVariable("diff", 0)

    problem += diff <= sum0 - sum1
    problem += diff <= sum1 - sum0

    problem += diff

    problem.solve(PULP_CBC_CMD(msg=False))

    subarrays = {
        node: y[node].varValue for node in graph.nodes()
    }

    cost = problem.objective.value()
    return subarrays, cost


def solve_bins(graph, bin_number):
    """
    Minimises the difference in the number of edges in each bin for subarrays 0 and 1.
    Args:
        graph: networkx graph of all antennas
        bin_number: number of bins

    Returns:
        subarrays: directory where each antenna is assigned a value of 0 or 1 depending on
                   which subarray it is in
        cost: sum of differences between the number of edges in each bin for each subarray
    """
    problem, y, z, n = setup(graph)

    bins = bin_maker(graph, bin_number)

    diff_vars = []
    for b in bins:
        masked_graph = graph_masker(graph, b[0], b[1])

        bin_sum0, bin_sum1 = sum_edges(masked_graph, y, z, n)

        diff = LpVariable(f"diff_{b[0]}_{b[1]}", 0)

        problem += diff >= bin_sum1 - bin_sum0
        problem += diff >= bin_sum0 - bin_sum1

        diff_vars.append(diff)

    problem += sum(diff_vars)
    problem.solve(PULP_CBC_CMD(msg=False))

    subarrays = {
        node: y[node].varValue for node in graph.nodes()
    }

    cost = problem.objective.value()
    return subarrays, cost


def solve_kl_divergence(graph, bin_number):
    """
    Minimises the kl divergence between subarrays 0 and 1.
    Args:
        graph: networkx graph of all antennas
        bin_number: number of bins

    Returns:
        subarrays: directory where each antenna is assigned a value of 0 or 1 depending on
                   which subarray it is in
        cost: final kl divergence
    """
    problem, y, z, n = setup(graph)

    bins = bin_maker(graph, bin_number)

    total_edges = len(graph.edges)

    kl_vars = []
    for b in bins:
        masked_graph = graph_masker(graph, b[0], b[1])

        bin_sum0, bin_sum1 = sum_edges(masked_graph, y, z, n)

        p = LpVariable(f"p0_{b[0]}_{b[1]}", 1e-4, 1)
        q = LpVariable(f"p1_{b[0]}_{b[1]}", 1e-4, 1)

        problem += p * total_edges == bin_sum0
        problem += q * total_edges == bin_sum1

        grid = np.linspace(1e-4, 1, 20)

        kl_p_bin = LpVariable(f"kl_{b[0]}_{b[1]}_p")

        kl_q_bin = LpVariable(f"kl_{b[0]}_{b[1]}_q")

        for i in grid:
            for j in grid:
                kl_ij = i * np.log(i / j)
                kl_ji = j * np.log(j / i)

                grad_ij_1 = np.log(i / j) + 1
                grad_ij_2 = -i / j

                grad_ji_1 = np.log(j / i) + 1
                grad_ji_2 = -j / i

                problem += kl_p_bin >= kl_ij + grad_ij_1 * (p - i) + grad_ij_2 * (q - j)
                problem += kl_q_bin >= kl_ji + grad_ji_1 * (q - j) + grad_ji_2 * (p - i)

        kl_vars.append(kl_p_bin)
        kl_vars.append(kl_q_bin)

    problem += sum(kl_vars)
    problem.solve(PULP_CBC_CMD(msg=False))

    subarrays = {}
    for i in graph.nodes():
        if y[i].varValue == 0:
            subarrays[i] = 0
        elif y[i].varValue == 1:
            subarrays[i] = 1

    cost = problem.objective.value()
    return subarrays, cost
