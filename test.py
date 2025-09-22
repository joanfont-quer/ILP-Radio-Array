import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from ska_ost_array_config import get_subarray_template
from visualisation import visualise_sol


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


def ska_mid_full_graph():
    aastar_core = get_subarray_template("Mid_full_AA4")
    antenna_names = aastar_core.array_config.names.data
    antenna_names = [str(name) for name in antenna_names]
    antenna_coords = aastar_core.array_config.xyz.values

    positions_3d = {name: tuple(coord) for name, coord in zip(antenna_names, antenna_coords)}
    positions_2d = {name: (coord[0], coord[1]) for name, coord in positions_3d.items()}
    graph = nx.complete_graph(antenna_names)

    # Assign graph weights by Euclidean distances between positions
    graph = euclidean_weight_assigner(graph, positions_3d)

    return graph, positions_3d, positions_2d


def plot_uv_coverage_partitioned(positions_3d, solution, frequency_hz):
    c = 299792458.0  # speed of light [m/s]
    wavelength = c / frequency_hz

    cmap = plt.get_cmap("tab10")

    uv_points = {label: [] for label in set(solution.values())}

    antennas = list(positions_3d.keys())
    for i, j in itertools.combinations(antennas, 2):
        xi, yi, zi = positions_3d[i]
        xj, yj, zj = positions_3d[j]

        dx, dy, dz = xi - xj, yi - yj, zi - zj
        u, v = dx / wavelength, dy / wavelength

        # Decide colour by subarray membership
        if solution[i] == solution[j]:
            uv_points[solution[i]].append((u, v))
            uv_points[solution[i]].append((-u, -v))

    # --- Plotting ---
    plt.figure(figsize=(7,7))

    # Same-size markers, coloured by subarray
    for label, points in uv_points.items():
        if len(points) == 0:
            continue
        pts = np.array(points)
        plt.scatter(
            pts[:,0], pts[:,1],
            s=10, alpha=0.8,
            color=cmap(label % 10),
            label=f"Subarray {label}"
        )

    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Snapshot uv-coverage by subarray")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig('/share/nas2_3/jfont/ILP-Radio-Array/Results/uv_coverage.png')


assignments_1 = get_subarray_template("Mid_split2_1_AA4")
assignments_2 = get_subarray_template("Mid_split2_2_AA4")

antenna_names_1 = assignments_1.array_config.names.data
antenna_names_1 = [str(name) for name in antenna_names_1]
antenna_coords_1 = assignments_1.array_config.xyz.values

positions_3d_1 = {name: tuple(coord) for name, coord in zip(antenna_names_1, antenna_coords_1)}
positions_2d_1 = {name: (coord[0], coord[1]) for name, coord in positions_3d_1.items()}
graph_1 = nx.complete_graph(antenna_names_1)

graph_1 = euclidean_weight_assigner(graph_1, positions_3d_1)


antenna_names_2 = assignments_2.array_config.names.data
antenna_names_2 = [str(name) for name in antenna_names_2]
antenna_coords_2 = assignments_2.array_config.xyz.values

positions_3d_2 = {name: tuple(coord) for name, coord in zip(antenna_names_2, antenna_coords_2)}
positions_2d_2 = {name: (coord[0], coord[1]) for name, coord in positions_3d_2.items()}
graph_2 = nx.complete_graph(antenna_names_2)

graph_2 = euclidean_weight_assigner(graph_2, positions_3d_2)

graph, positions_3d, positions_2d = ska_mid_full_graph()


solution_1 = {node: 0 for node in graph_1.nodes()}
solution_2 = {node: 1 for node in graph_2.nodes()}

solution = {**solution_1, **solution_2}


plot_uv_coverage_partitioned(positions_3d, solution, frequency_hz=1.4e9)
