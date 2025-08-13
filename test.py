from pathlib import Path
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

METADATA_FILE = Path("metadata.parquet")
SOLUTIONS_FILE = Path("solutions.npz")

df = pd.read_parquet(METADATA_FILE)
print(df)

