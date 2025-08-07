from gurobipy import GRB, quicksum, Model
from Partitioner.utils import *


def setup(graph, subarray_number):
    """
    Sets up model and decision variables in the problem.
    Args:
        graph: Networkx graph of all antennas.
        subarray_number: Number of subarrays we want to split the graph into.

    Returns:
        model: Initialised gurobi model.
        y: Dictionary of binary variables y[i, s] is 1 is antenna 'i' is in subarray 's', 0 otherwise.
        node_list: List of antenna names
    """
    node_list = list(graph.nodes)

    model = Model("RadioArrayDivider")

    model.setParam("OutputFlag", 1)

    y = model.addVars(node_list, range(subarray_number), vtype=GRB.BINARY, name="y")

    # antenna 'i' can only be assigned to one subarray.
    model.addConstrs(
        (quicksum(y[i, s] for s in range(subarray_number)) == 1 for i in node_list), name="node_constraint"
    )

    # forces the first antenna to be in the first subarray, which breaks subarray symmetry.
    model.addConstr(y[node_list[0], 0] == 1, name="symmetry_break_constraint")

    # each subarray should have at least one baseline
    model.addConstrs(quicksum(y[i, s] for i in node_list) >= 2 for s in range(subarray_number))
    return model, y, node_list


def sum_edges(graph, model, y, b, subarray_number):
    """
    Computes the sum of active edges within each subarray.
    Args:
        graph: Networkx graph of all antennas, with edge weights either 1 if active or 0 if inactive.
        model: Gurobi model.
        y: Dictionary of binary variables y[i, s] is 1 is antenna 'i' is in subarray 's', 0 otherwise.
        b: Bin being processed. b = [a, b], where a and b are where the bin starts and finishes respectively.
        subarray_number: Number of subarrays we want to split the graph into.

    Returns:
        sum_vars: Dictionary of variables representing the sum of active edges within each subarray.
    """
    active_edges = [(i, j) for i, j, d in graph.edges(data=True) if d['weight'] > 0]

    sum_vars = model.addVars(range(subarray_number), lb=0, name=f"sum_var_{b[0]}_{b[1]}")

    model.addConstrs(
        (sum_vars[s] == quicksum(y[i, s] * y[j, s] for i, j in active_edges)
         for s in range(subarray_number)),
        name=f"edge_number_sum_{b[0]}_{b[1]}"
    )

    return sum_vars


def solve_pdf_bins(graph, bin_number, subarray_number):
    """
    Solves the optimisation problem of partitioning the graph into subarrays where all subarrays have as close to the
    same number of nodes in each baseline bin as possible.
    Args:
        graph: Networkx graph of all antennas.
        bin_number: Number of bins we are partitioning the baselines into.
        subarray_number: Number of subarrays we want to split the graph into.

    Returns:
        subarrays: Dictionary mapping each node to its assigned subarray.
        cost: Final objective value of the solution.
    """
    model, y, node_list = setup(graph, subarray_number)
    bins = quantile_bin_maker(graph, bin_number)

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
    model.setParam("Cuts", 3)
    model.optimize()

    subarrays = {}
    for i in node_list:
        for s in range(subarray_number):
            if y[(i, s)].X > 0.5:
                subarrays[i] = s

    cost = model.ObjVal
    return subarrays, cost


def solve_cdf_bins(graph, bin_number, subarray_number):
    model, y, node_list = setup(graph, subarray_number)
    bins = quantile_bin_maker(graph, bin_number)

    diff_vars = []
    cdf_sum_previous = 0
    for b in bins:
        masked_graph = graph_masker(graph, b[0], b[1])

        bin_sums = sum_edges(masked_graph, model, y, b, subarray_number)

        sum_max = model.addVar(lb=0, name=f"sum_max_{b[0]}_{b[1]}")
        sum_min = model.addVar(lb=0, name=f"sum_min_{b[0]}_{b[1]}")

        if cdf_sum_previous == 0:
            model.addGenConstrMax(sum_max, bin_sums, name=f"max_constr_{b[0]}_{b[1]}")
            model.addGenConstrMin(sum_min, bin_sums, name=f"min_constr_{b[0]}_{b[1]}")
            cdf_sum_previous = bin_sums
        else:
            cdf_sum = model.addVars(range(subarray_number), lb=0, name=f"cdf_{b[0]}_{b[1]}")
            model.addConstrs((cdf_sum[i] == cdf_sum_previous[i] + bin_sums[i] for i in range(subarray_number)),
                            name=f"cdf_def_{b[0]}_{b[1]}")

            model.addGenConstrMax(sum_max, cdf_sum, name=f"max_constr_{b[0]}_{b[1]}")
            model.addGenConstrMin(sum_min, cdf_sum, name=f"min_constr_{b[0]}_{b[1]}")
            cdf_sum_previous = cdf_sum

        diff = model.addVar(lb=0, name=f"diff_{b[0]}_{b[1]}")

        model.addConstr(diff >= sum_max - sum_min,
                        name=f"diff_constr_{b[0]}_{b[1]}")
        diff_vars.append(diff)
    model.update()

    model.setObjective(quicksum(diff_vars), GRB.MINIMIZE)
    model.setParam("Threads", 8)
    model.setParam("Cuts", 3)
    model.optimize()

    subarrays = {}
    for i in node_list:
        for s in range(subarray_number):
            if y[(i, s)].X > 0.5:
                subarrays[i] = s

    cost = model.ObjVal
    return subarrays, cost


def solve_pdf_bins_lambda(graph, bin_number, subarray_number, frequency_list):
    model, y, node_list = setup(graph, subarray_number)
    bins = quantile_bin_maker_lambda(graph, bin_number, frequency_list)

    diff_vars = []
    for b in bins:
        sum_vars = model.addVars(range(subarray_number), lb=0, name=f"sum_var_{b[0]}_{b[1]}")
        for s in range(subarray_number):
            freq = frequency_list[s]
            masked_graph = graph_masker_lambda(graph, freq, b[0], b[1])
            active_edges = [(i, j) for i, j, d in masked_graph.edges(data=True) if d['weight'] > 0]
            print(active_edges)

            model.addConstr(sum_vars[s] == quicksum(y[i, s] * y[j, s] for i, j in active_edges),
                            name=f"edge_number_sum_{b[0]}_{b[1]}_{s}")


        sum_max = model.addVar(lb=0, name=f"sum_max_{b[0]}_{b[1]}")
        sum_min = model.addVar(lb=0, name=f"sum_min_{b[0]}_{b[1]}")
        model.addGenConstrMax(sum_max, list(sum_vars.values()), name=f"max_constr_{b[0]}_{b[1]}")
        model.addGenConstrMin(sum_min, list(sum_vars.values()), name=f"min_constr_{b[0]}_{b[1]}")

        diff = model.addVar(lb=0, name=f"diff_{b[0]}_{b[1]}")

        model.addConstr(diff >= sum_max - sum_min, name=f"diff_constr_{b[0]}_{b[1]}")
        diff_vars.append(diff)
    model.update()

    model.setObjective(quicksum(diff_vars), GRB.MINIMIZE)
    model.setParam("Threads", 8)
    model.setParam("Cuts", 3)
    model.optimize()

    subarrays = {}
    for i in node_list:
        for s in range(subarray_number):
            if y[(i, s)].X > 0.5:
                subarrays[i] = s

    cost = model.ObjVal
    return subarrays, cost
