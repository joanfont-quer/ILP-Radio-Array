import numpy as np
import networkx as nx


def euclidean_distance(u, v):
    """
    Calculates the Euclidean distance between two positions.
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


def read_tsp_file(file_name):
    """
    Generates a graph from a given file with positional data for the nodes.
    Args:
        file_name: Name of the file.

    returns:
        graph: networkx graph
    """
    with open(file_name) as f:
        lines = f.read().strip().split('\n')

    positions = {}
    for i, line in enumerate(lines):
        if line.startswith("NODE_COORD_SECTION"):
            display_data_section = lines[i + 1:]
            for entry in display_data_section:
                if entry.strip() == "EOF":
                    break
                parts = entry.split()
                node = int(parts[0]) - 1
                x, y = map(float, parts[1:])
                positions[node] = (x, y)

    n = len(positions)
    graph = nx.complete_graph(n)

    # Assign graph weights by Euclidean distances between positions
    graph = euclidean_weight_assigner(graph, positions)
    return graph, positions


def generate_graph(node_num, seed=np.random.randint(1, 10000)):
    """
    Generate a random graph in Euclidean space given the number of nodes
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
