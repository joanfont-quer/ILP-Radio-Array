import numpy as np
import networkx as nx
from ska_ost_array_config import get_subarray_template


def euclidean_distance(u, v):
    """
    Calculates the Euclidean distance between two positions using einsum.
    Args:
        u: position one
        v: position two

    Returns:
        dist: Euclidean distance between u and v
    """
    u = np.array(u)
    v = np.array(v)

    diff = u - v
    dist = np.sqrt(np.einsum('i,i->', diff, diff))
    return dist


def euclidean_weight_assigner(graph, positions):
    """
    Assign weights to graph edges according to the Euclidean distance between nodes.
    Args:
        graph: networkx graph without edge weights
        positions: list of node positions in the x and y coordinates

    Returns:
        graph: networkx graph with edge weights
    """
    for i, j in graph.edges:
        # Calculate Euclidean distance between nodes i and j
        dist = euclidean_distance(positions[i], positions[j])
        graph[i][j]['weight'] = dist
    return graph


def generate_graph(node_num, seed=np.random.randint(1, 10000)):
    """
    Generate a random graph in Euclidean space given the number of nodes.

    Args:
        node_num: number of nodes the graph should have
        seed: numpy seed for reproducibility, if no seed is provided, generate a random seed

    Returns:
        graph: networkx graph
        positions: list of node positions in the x and y coordinates
    """
    np.random.seed(seed)
    positions = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100))
                 for i in range(node_num)}
    graph = nx.complete_graph(node_num)

    # Assign graph weights by Euclidean distances between positions
    graph = euclidean_weight_assigner(graph, positions)

    return graph, positions


def generate_graph_gaussian(node_num, seed=np.random.randint(1, 10000)):
    """
    Generate a random graph in Euclidean space with the nodes distributed according to a Gaussian.

    Args:
        node_num (int): Desired number of nodes in the graph.
        seed (int, optional): Random seed to utilise. Defaults to np.random.randint(1, 10000).

    Returns:
        graph: networkx graph
        positions: list of node positions in the x and y coordinates
    """
    np.random.seed(seed)
    positions = {i: (np.random.normal(0, 1), np.random.normal(0, 1))
                 for i in range(node_num)}
    graph = nx.complete_graph(node_num)

    # Assign graph weights by Euclidean distances between positions
    graph = euclidean_weight_assigner(graph, positions)

    return graph, positions


def ska_mid_graph_from_template(template_name):
    subarray = get_subarray_template(template_name)
    antenna_names = [str(name) for name in subarray.array_config.names.data]
    antenna_coords = subarray.array_config.xyz.values

    positions_3d = {name: tuple(coord) for name, coord in zip(antenna_names, antenna_coords)}
    positions_2d = {name: (coord[0], coord[1]) for name, coord in positions_3d.items()}

    graph = nx.complete_graph(antenna_names)
    graph = euclidean_weight_assigner(graph, positions_3d)

    return graph, positions_3d, positions_2d


def ska_mid_graph():
    return ska_mid_graph_from_template("MID_INNER_R1KM_AASTAR")


def ska_mid_full_graph():
    return ska_mid_graph_from_template("Mid_full_AA4")
