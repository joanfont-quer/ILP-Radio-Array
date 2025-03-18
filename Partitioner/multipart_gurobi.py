from gurobipy import GRB, quicksum, Model
from Partitioner.utils import *


def setup(graph, subarray_number):
    """
    Sets up used variables in the problem.
    Args:
        graph: Networkx graph of all antennas.
        subarray_number: Number of subarrays we want to split the graph into.

    Returns:
        model: Initialised gurobi model.
        y: Dictionary of binary variables y[i, s] is 1 is antenna 'i' is in subarray 's', 0 otherwise.
        n: Number of antennas.
    """
    n = len(graph.nodes)

    model = Model("RadioArrayDivider")

    model.setParam("OutputFlag", 0)

    y = model.addVars(n, subarray_number, vtype=GRB.BINARY, name="y")

    # antenna 'i' can only be assigned to one subarray.
    model.addConstrs(
        (quicksum(y[i, s] for s in range(subarray_number)) == 1 for i in range(n)),
        name="node_constraint"
    )

    # forces the first antenna to be in the first subarray, which breaks subarray symmetry.
    model.addConstr(y[0, 0] == 1, name="symmetry_break_constraint")

    # each subarray should have at least one baseline
    model.addConstrs(quicksum(y[i, s] for i in range(n)) >= 2 for s in range(subarray_number))

    model.update()
    return model, y, n


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


def sum_baselines_sq(graph, model, y, subarray_number):
    sum_vars = model.addVars(range(subarray_number), lb=0, name="sum_vars")

    model.addConstrs(
        (sum_vars[s] == quicksum((graph[i][j]["weight"] ** 2) * y[i, s] * y[j, s]
                                             for i, j in graph.edges())
         for s in range(subarray_number)), name="squared_baseline_sums")
    return sum_vars


def solve_bins(graph, bin_number, subarray_number):
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


def solve_kl(graph, subarray_number):
    model, y, n = setup(graph, subarray_number)

    d_sums = sum_baselines_sq(graph, model, y, subarray_number)
    num_edge_sums = sum_edges(graph, model, y, [1, 1], subarray_number)

    scale_parameters = model.addVars(range(subarray_number), lb=0, name="sigma_mll")

    model.addConstrs((2 * num_edge_sums[s] * scale_parameters[s] * scale_parameters[s] == d_sums[s]
                     for s in range(subarray_number)), name="sigma_mll_definition")

    max_weight = max(data["weight"] for i, j, data in graph.edges(data=True))
    scale_parameters_upper_bound = max_weight / np.sqrt(2)
    sigma_points = np.linspace(1e-4, scale_parameters_upper_bound, 100)
    ln_points = [np.log(x) for x in sigma_points]

    log_scale_params = model.addVars(range(subarray_number), lb=-GRB.INFINITY, name="log_sigma_mll")
    for s in range(subarray_number):
        model.addGenConstrPWL(scale_parameters[s], log_scale_params[s],
                               sigma_points.tolist(), ln_points,
                               name="pwl_log_sigma")

    nll_terms = []
    for s in range(subarray_number):
        log_sum = quicksum(np.log(graph[i][j]["weight"]) * y[i, s] * y[j, s] for i, j in graph.edges())
        nll_terms.append(2 * num_edge_sums[s] * log_scale_params[s] + num_edge_sums[s] - log_sum)

    nll_total = quicksum(nll_terms)

    sigma_max = model.addVar(lb=0, name="sigma_max")
    sigma_min = model.addVar(lb=0, name="sigma_min")
    model.addGenConstrMax(sigma_max, scale_parameters, name=f"sigma_max_constr")
    model.addGenConstrMin(sigma_min, scale_parameters, name=f"sigma_min_constr")

    diff = model.addVar(lb=0, name="diff")
    model.addConstr(diff >= sigma_max - sigma_min, name=f"diff_constr")
    model.update()

    model.setObjective(diff + nll_total, GRB.MINIMIZE)
    model.setParam("Threads", 8)
    model.optimize()

    subarrays = {}
    for i in range(n):
        for s in range(subarray_number):
            if y[(i, s)].X > 0.5:
                subarrays[i] = s

    cost = model.ObjVal
    return subarrays, cost
