from pathlib import Path
import numpy as np
from utils import build_subarrays_from_assignments
import pandas as pd

HERE = Path(__file__).resolve().parent

NAME = "heuristic"
SOLUTION_PATH = HERE.parent / "solution_files" / "solutions_heuristic_ska_full.npz"
METADATA_PATH = HERE.parent / "metadata_files" / "metadata_heuristic_1.parquet"
SOLUTION_KEY = "node197_sub2_seed137"
OUTPUT_DIR = "plots/"

def get_subarrays():
    solutions = dict(np.load(SOLUTION_PATH, allow_pickle=True))
    sol = solutions[SOLUTION_KEY].item()
    subarrays = build_subarrays_from_assignments(sol)
    return subarrays


def metadata():
    return pd.read_parquet(METADATA_PATH, engine="fastparquet")


get_subarrays()
