import numpy as np
import networkx as nx
import scipy

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


def quantile_bin_maker(graph, bin_number):
    weight_list = np.fromiter(map(lambda e: e[2]['weight'], graph.edges(data=True)), dtype=float)

    sorted_weights = np.sort(weight_list)

    quantiles = np.linspace(0, 1, bin_number + 1)
    bin_edges = np.quantile(sorted_weights, quantiles)

    bin_edges = np.unique(bin_edges)
    if len(bin_edges) - 1 < bin_number:
        raise ValueError("Too few distinct weights to form the requested number of equal-sized bins.")

    bins = list(zip(bin_edges[:-1], bin_edges[1:]))
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


def graph_masker_lambda(graph, frequency, a=0, b=np.inf):
    c = scipy.constants.c

    masked_graph = nx.Graph()
    masked_graph.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        baseline_lambda = (d['weight'] * frequency) / c
        new_weight = 1 if a <= baseline_lambda <= b else 0

        masked_graph.add_edge(u, v, weight=new_weight)

    return masked_graph


def quantile_bin_maker_lambda(graph, bin_number, frequencies):
    c = scipy.constants.c
    weight_list_m = np.fromiter(map(lambda e: e[2]['weight'], graph.edges(data=True)), dtype=float)

    lambda_weights = []
    for f in frequencies:
        lambda_weights.extend((weight_list_m * f) / c)

    lambda_weights = np.array(lambda_weights)
    sorted_weights = np.sort(lambda_weights)

    quantiles = np.linspace(0, 1, bin_number + 1)
    bin_edges = np.quantile(sorted_weights, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) - 1 < bin_number:
        raise ValueError("Too few distinct λ-space weights to form requested number of bins.")

    bins = list(zip(bin_edges[:-1], bin_edges[1:]))
    return bins
