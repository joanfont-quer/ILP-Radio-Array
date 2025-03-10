import numpy as np
import networkx as nx


def euclidean_distance(u, v):
    x1, y1 = u
    x2, y2 = v

    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def euclidean_weight_assigner(graph, positions):
    # Assign graph weights by Eucledian distances between positions
    for i, j in graph.edges:
        # Calculate Euclidean distance between nodes i and j
        dist = euclidean_distance(positions[i], positions[j])
        graph[i][j]['weight'] = dist
    return graph


def read_tsp_file(file_name):
    """
    Generates a graph from a given file with positional data for the nodes.
    parameters:
        file_name (str): Name of the file.
    returns:
        graph (networkx.Graph): A graph with nodes and weighted edges.
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

    # Assign graph weights by Eucledian distances between positions
    graph = euclidean_weight_assigner(graph, positions)
    return graph, positions


def generate_graph(city_num, seed):
    np.random.seed(seed)
    positions = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100))
                 for i in range(city_num)}
    graph = nx.complete_graph(city_num)

    # Assign graph weights by Eucledian distances between positions
    graph = euclidean_weight_assigner(graph, positions)

    return graph, positions
