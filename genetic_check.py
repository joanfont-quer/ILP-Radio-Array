import pickle as pkl
from ska_ost_array_config import get_subarray_template
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import graph_loader as gl
import visualisation as vis
import bisect
from Partitioner.multipart_heuristic import wasserstein
from visualisation import visualise_sol


def plot_ga_progress(result):
    if not hasattr(result, "history"):
        print("No history found in the result — was save_history=True?")
        return

    gens = []
    f_min = []
    f_avg = []
    f_std = []

    for algo in result.history:
        gens.append(algo.n_gen)
        pop = algo.pop
        F = pop.get("F")

        f_min.append(np.min(F))
        f_avg.append(np.mean(F))
        f_std.append(np.std(F))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(gens, f_min, label="Best objective", color="tab:blue")
    ax[0].set_xlabel("Generation")
    ax[0].set_ylabel("Best objective")
    ax[0].tick_params(axis="y", labelcolor="tab:blue")

    ax[1].plot(gens, f_avg, label="Average objective", color="tab:orange")
    ax[1].fill_between(gens, np.array(f_avg) - np.array(f_std),
                     np.array(f_avg) + np.array(f_std),
                     color="tab:orange", alpha=0.2, label="Std. deviation")
    ax[1].set_xlabel("Generation")
    ax[1].set_ylabel("Average objective")
    ax[1].set_yscale("log")
    ax[1].tick_params(axis="y", labelcolor="tab:orange")

    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    with open("ga_result.pkl", "rb") as f:
        result = pkl.load(f)

    graph, positions_3d, positions_2d = gl.ska_mid_full_graph()
    nodes = list(graph.nodes())
    print(type(result))
    x = result.X
    f = result.F
    print(f)
    solution_dict = {node: part for node, part in zip(nodes, x)}
    vis.visualise_sol(solution_dict, positions_2d, graph)

    plot_ga_progress(result)

    assignments_1 = get_subarray_template("Mid_split2_1_AA4")
    assignments_2 = get_subarray_template("Mid_split2_2_AA4")

    antenna_names_1 = assignments_1.array_config.names.data
    antenna_names_1 = [str(name) for name in antenna_names_1]
    antenna_coords_1 = assignments_1.array_config.xyz.values

    positions_3d_1 = {name: tuple(coord) for name, coord in zip(antenna_names_1, antenna_coords_1)}
    graph_1 = nx.complete_graph(antenna_names_1)

    graph_1 = gl.euclidean_weight_assigner(graph_1, positions_3d_1)

    antenna_names_2 = assignments_2.array_config.names.data
    antenna_names_2 = [str(name) for name in antenna_names_2]
    antenna_coords_2 = assignments_2.array_config.xyz.values

    positions_3d_2 = {name: tuple(coord) for name, coord in zip(antenna_names_2, antenna_coords_2)}
    graph_2 = nx.complete_graph(antenna_names_2)

    graph_2 = gl.euclidean_weight_assigner(graph_2, positions_3d_2)

    solution_1 = {node: 0 for node in graph_1.nodes()}
    solution_2 = {node: 1 for node in graph_2.nodes()}

    solution = {**solution_1, **solution_2}

    visualise_sol(solution, positions_2d, graph)

    graph, positions_3d, positions_2d = gl.ska_mid_full_graph()

    partition_number = 2
    partition_weights = {p: [] for p in range(partition_number)}

    for u, v, data in graph.edges(data=True):
        p_u = int(solution[u])
        p_v = int(solution[v])
        if p_u == p_v:
            bisect.insort(partition_weights[p_u], data['weight'])

    cost = wasserstein(partition_weights, {"p": 1, "alpha": 1})
    print(cost)


if __name__ == "__main__":
    main()
