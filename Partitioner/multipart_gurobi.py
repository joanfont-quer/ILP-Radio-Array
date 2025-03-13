from gurobipy import GRB, quicksum, Model
from Partitioner.utils import *


def setup(graph, subarray_number):
    n = len(graph.nodes)

    model = Model("RadioArrayDivider")

    model.setParam("OutputFlag", 0)

    y = model.addVars(n, subarray_number, vtype=GRB.BINARY, name="y")

    # antenna 'i' can only be assigned to one subarray
    model.addConstrs(
        (quicksum(y[i, s] for s in range(subarray_number)) == 1 for i in range(n)),
        name="node_constraint"
    )

    # forces the first antenna to be in the first subarray, which breaks the subarray symmetry
    model.addConstr(y[0, 0] == 1, name="symmetry_break_constraint")

    model.update()
    return model, y, n


def sum_edges(graph, model, y, b, subarray_number):
    active_edges = [(i, j) for i, j, d in graph.edges(data=True) if d['weight'] > 0]

    sum_vars = model.addVars(range(subarray_number), lb=0, name=f"sum_var_{b[0]}_{b[1]}")

    model.addConstrs(
        (sum_vars[s] == quicksum(y[i, s] * y[j, s] for i, j in active_edges)
         for s in range(subarray_number)),
        name=f"link_sum_{b[0]}_{b[1]}"
    )

    return sum_vars


def solve_bins(graph, bin_number, subarray_number):
    model, y, n = setup(graph, subarray_number)
    bins = bin_maker(graph, bin_number)

    diff_vars = []
    for b in bins:
        masked_graph = graph_masker(graph, b[0], b[1])

        bin_sums = sum_edges(masked_graph, model, y, b, subarray_number)

        sum_max = model.addVar(lb=0, name=f"sum_max_{b[0]}_{b[1]}")
        sum_min = model.addVar(lb=0, name=f"sum_min_{b[0]}_{b[1]}")
        model.addGenConstrMax(sum_max, bin_sums, name=f"max_constr_{b[0]}_{b[1]}")
        model.addGenConstrMin(sum_min, bin_sums, name=f"min_constr_{b[0]}_{b[1]}")

        diff = model.addVar(lb=0, name=f"diff_{b[0]}_{b[1]}")

        model.addConstr(diff >= sum_max - sum_min,
                        name=f"diff_constr_{b[0]}_{b[1]}")
        diff_vars.append(diff)
    model.update()

    model.setObjective(quicksum(diff_vars), GRB.MINIMIZE)
    model.setParam("Threads", 8)
    model.optimize()

    subarrays = {}
    for i in range(n):
        for s in range(subarray_number):
            if y[(i, s)].X > 0.5:
                subarrays[i] = s

    cost = model.ObjVal
    return subarrays, cost
