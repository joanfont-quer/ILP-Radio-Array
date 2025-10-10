from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

NAME = "heuristic"
SOLUTION_PATH = HERE.parent / "solution_files" / "solutions_combined.npz"
METADATA_PATH = HERE.parent / "metadata_files" / "metadata_combined.parquet"

OUTPUT_DIR = "plots/"

def metadata():
    return pd.read_parquet(METADATA_PATH, engine="fastparquet")
