from pathlib import Path
from utils import build_subarrays_from_assignments
import pickle as pkl
import graph_loader as gl

NAME = "genetic"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "ga_result.pkl"


def get_subarrays():
    with open(RESULTS, "rb") as f:
        result = pkl.load(f)

    graph, positions_3d, positions_2d = gl.ska_mid_full_graph()
    nodes = list(graph.nodes())

    x = result.X

    solution_dict = {node: part for node, part in zip(nodes, x)}

    subarrays = build_subarrays_from_assignments(solution_dict)
    return subarrays
