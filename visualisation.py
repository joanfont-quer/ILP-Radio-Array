from Partitioner.utils import bin_maker, graph_masker
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time


def visualise_sol(solution, positions, graph):
    cmap = plt.get_cmap("tab10")
    colour_map = [cmap(solution[node] % 10) for node in graph.nodes()]

    nx.draw(graph, positions, node_color=colour_map, with_labels=False, node_size=250, edgelist=[])
    nx.draw_networkx_labels(graph, positions, font_color='white')
    plt.savefig(f"Results/graph_partition_{time.time()}.png")


def visualise_bins(solution, graph, bin_number, subarray_number):
    bins = bin_maker(graph, bin_number)

    bin_counts = {s: [] for s in range(subarray_number)}

    for a, b in bins:
        masked_graph = graph_masker(graph, a, b)
        for s in range(subarray_number):
            count_s = sum(
                1 for u, v, d in masked_graph.edges(data=True)
                if d['weight'] == 1 and solution[u] == s and solution[v] == s
            )
            bin_counts[s].append(count_s)

    bin_labels = [f"{low:.2f}-{high:.2f}" for low, high in bins]
    x = np.arange(len(bins))
    width = 0.8 / subarray_number

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for s in range(subarray_number):
        plt.bar(x + s * width, bin_counts[s], width=width, label=f"Subarray {s}",
                color=cmap(s % 10), alpha=0.7)

    plt.xlabel("Baseline Bins")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Edges in Each Baseline Category per Subarray")
    plt.xticks(x + width * (subarray_number - 1) / 2, bin_labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(f"Results/baseline_histogram_{time.time()}.png")
