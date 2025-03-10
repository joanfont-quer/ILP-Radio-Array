from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import numpy as np
import networkx as nx


def setup(graph):
    n = len(graph.nodes)
    problem = LpProblem("RadioArrayDivider", LpMinimize)

    # y_i is 1 if antenna 'i' is in subarray 1, 0 if antenna 'i' is in subarray 0
    y = {
        i: LpVariable(f"y_{i}", 0, 1, cat="Binary")
        for i in range(n)
    }

    # auxiliary variable equal to y_i * y_j
    z = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                z[(i,j)] = LpVariable(f"z_{i}_{j}", 0, 1, cat="Binary")
                problem += z[(i,j)] <= y[i] # if y_i is 0 then z is 0
                problem += z[(i,j)] <= y[j] # if y_j is 0 then z is 0
                problem += z[(i,j)] >= y[i] + y[j] - 1 # if both y_i and y_j are 1 then z is 1

    return problem, y, z, n


def sum_edges(graph, y, z, n):
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
    return subarrays, problem.objective.value()


def bin_maker(graph, bin_number):
    weight_list = np.fromiter(map(lambda e: e[2]['weight'], graph.edges(data=True)), dtype=float)

    max_weight = np.max(weight_list)
    min_weight = np.min(weight_list)

    bin_borders = np.linspace(min_weight, max_weight, bin_number)

    bins = list(zip(bin_borders[:-1], bin_borders[1:]))
    return bins


def graph_masker(graph, a, b):

    masked_graph = nx.Graph()
    masked_graph.add_nodes_from(graph.nodes())

    for u, v, d in graph.edges(data=True):
        weight = d['weight']
        new_weight = 1 if a <= weight <= b else 0

        masked_graph.add_edge(u, v, weight=new_weight)

    return masked_graph


def solve_bins(graph, bin_number):
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

    problem += lpSum(diff_vars)
    problem.solve(PULP_CBC_CMD(msg=False))

    subarrays = {
        node: y[node].varValue for node in graph.nodes()
    }
    return subarrays, problem.objective.value()


def solve_kl_divergence(graph, bin_number):
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

        kl_p = LpVariable(f"kl_{b[0]}_{b[1]}_p", 0)

        kl_q = LpVariable(f"kl_{b[0]}_{b[1]}_q", 0)

        for i in grid:
            for j in grid:
                kl_ij = i * np.log(i / j)
                kl_ji = j * np.log(j / i)

                grad_ij_1 = np.log(i / j) + 1
                grad_ij_2 = -i / j

                grad_ji_1 = np.log(j / i) + 1
                grad_ji_2 = -j / i

                problem += kl_p >= kl_ij + grad_ij_1 * (p - i) + grad_ij_2 * (q - j)
                problem += kl_q >= kl_ji + grad_ji_1 * (q - j) + grad_ji_2 * (p - i)

        kl_vars.append(kl_p)
        kl_vars.append(kl_q)

    problem += lpSum(kl_vars)
    problem.solve(PULP_CBC_CMD(msg=False))

    subarrays = {
        node: y[node].varValue for node in graph.nodes()
    }
    return subarrays, problem.objective.value()

