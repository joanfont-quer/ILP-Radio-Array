from bipartitioner import bin_maker, graph_masker
import networkx as nx
import matplotlib.pyplot as plt


def visualise_sol(solution, positions, graph):
    colour_map = ['red' if solution[node] == 0 else 'blue' for node in graph.nodes()]

    nx.draw(graph, positions, node_color=colour_map, with_labels=False, node_size=250, edgelist=[])
    nx.draw_networkx_labels(graph, positions, font_color='white')
    plt.show()


def visualise_bins(solution, graph, bin_number):
    bins = bin_maker(graph, bin_number)

    bin_counts_0 = []
    bin_counts_1 = []

    for a, b in bins:
        masked_graph = graph_masker(graph, a, b)

        counts_array_0 = sum(1 for u, v, d in masked_graph.edges(data=True)
                             if d['weight'] == 1 and int(solution[u]) == int(solution[v]) == 0
                             )

        counts_array_1 = sum(1 for u, v, d in masked_graph.edges(data=True)
                             if d['weight'] == 1 and int(solution[u]) == int(solution[v]) == 1
                             )

        bin_counts_0.append(counts_array_0)
        bin_counts_1.append(counts_array_1)

    bin_labels = [f"{low:.2f}-{high:.2f}" for low, high in bins]
    x = range(len(bins))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x, bin_counts_0, width=width, label="Subarray 0", alpha=0.7)
    plt.bar([i + width for i in x], bin_counts_1, width=width, label="Subarray 1",
            alpha=0.7)

    plt.xlabel("Baseline Bins")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Edges in Each Baseline Category per Subarray")
    plt.xticks([i + width / 2 for i in x], bin_labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()