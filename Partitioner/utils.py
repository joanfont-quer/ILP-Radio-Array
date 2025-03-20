import numpy as np
import networkx as nx


def bin_maker(graph, bin_number):
    """
    Bin creator.
    Args:
        graph: networkx graph
        bin_number: number of bins

    Returns:
        bins: list of the edges of each bin
    """
    weight_list = np.fromiter(map(lambda e: e[2]['weight'], graph.edges(data=True)), dtype=float)

    max_weight = np.max(weight_list)
    min_weight = np.min(weight_list)

    bin_borders = np.linspace(min_weight, max_weight, bin_number + 1)

    bins = list(zip(bin_borders[:-1], bin_borders[1:]))
    return bins


def graph_masker(graph, a=0, b=np.inf):
    """
    Creates a mask of a graph where the edge weights are 1 if the edge is in the bin (a, b)
    and 0 otherwise.
    Args:
        graph: networkx graph
        a: lower bound for the bin
        b: upper bound for the bin

    Returns:
        masked_graph: masked networkx graph
    """
    masked_graph = nx.Graph()
    masked_graph.add_nodes_from(graph.nodes())

    for u, v, d in graph.edges(data=True):
        weight = d['weight']
        new_weight = 1 if a <= weight <= b else 0

        masked_graph.add_edge(u, v, weight=new_weight)

    return masked_graph
