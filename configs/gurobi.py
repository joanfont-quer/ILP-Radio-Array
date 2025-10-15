from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent

NAME = "gurobi"
SOLUTION_PATH = HERE.parent / "solution_files" / "solutions_combined.npz"
METADATA_PATH = HERE.parent / "metadata_files" / "metadata_combined.parquet"

OUTPUT_DIR = "plots/"

def metadata():
    return pd.read_parquet(METADATA_PATH, engine="fastparquet")


def solutions():
    data = dict(np.load(SOLUTION_PATH, allow_pickle=True))
    unpacked = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            unpacked[key] = value.item()
        else:
            unpacked[key] = value
    return unpacked
