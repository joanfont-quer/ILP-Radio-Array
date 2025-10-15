from pathlib import Path
from utils import build_subarrays_from_assignments
from Partitioner.multipart_evolutionary import GeneticAlgorithm
import Partitioner.objectives as obj
import pickle as pkl
import graph_loader as gl

NAME = "genetic"
HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE.parent / "ga_result.pkl"

OBJECTIVE = obj.wasserstein
OBJECTIVE_PARAMETERS = {"p": 1, "alpha": 1}
POP_SIZE = 100
N_GEN = 650


def solver(graph, seed, subarray_num):
    ga = GeneticAlgorithm(graph, subarray_num, OBJECTIVE, objective_parameters=OBJECTIVE_PARAMETERS,
                 pop_size=POP_SIZE, n_gen=N_GEN, seed=seed)
    return ga


def get_subarrays():
    solution_dict = get_ska_solution()

    subarrays = build_subarrays_from_assignments(solution_dict)
    return subarrays


def get_ska_solution():
    with open(RESULTS_FILE, "rb") as f:
        result = pkl.load(f)

    graph, positions_3d, positions_2d = gl.ska_mid_full_graph()
    nodes = list(graph.nodes())

    x = result.X

    solution_dict = {node: part for node, part in zip(nodes, x)}
    return solution_dict
