from pathlib import Path
import numpy as np
from utils import build_subarrays_from_assignments
from Partitioner.multipart_heuristic import HeuristicGraphPartitioner
import Partitioner.objectives as obj
import pandas as pd

NAME = "heuristic"
HERE = Path(__file__).resolve().parent

OBJECTIVE = obj.wasserstein
OBJECTIVE_PARAMETERS = {"p": 1, "alpha": 1}

SOLUTION_PATH = HERE.parent / "solution_files" / "solutions_heuristic_1.npz"
METADATA_PATH = HERE.parent / "metadata_files" / "metadata_heuristic_1.parquet"
SOLUTION_KEY = "node197_sub2_seed137"
OUTPUT_DIR = "plots/"

def solver(graph, seed, subarray_num):
    hgp = HeuristicGraphPartitioner(graph, subarray_num, OBJECTIVE, OBJECTIVE_PARAMETERS, seed=seed)
    return hgp


def get_subarrays():
    solutions = dict(np.load(HERE.parent / "solution_files" / "solutions_heuristic_ska_full.npz", allow_pickle=True))
    sol = solutions[SOLUTION_KEY].item()
    subarrays = build_subarrays_from_assignments(sol)
    return subarrays


def metadata():
    return pd.read_parquet(METADATA_PATH, engine="fastparquet")


def solutions():
    data = dict(np.load(SOLUTION_PATH, allow_pickle=True))
    unpacked = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # convert to dict: node index -> partition id
            unpacked[key] = {i: int(v) for i, v in enumerate(value)}
        else:
            unpacked[key] = value
    return unpacked
