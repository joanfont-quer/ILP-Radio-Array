from Partitioner.multipart_heuristic import solver
import graph_loader as gl
import numpy as np
import pandas as pd
import time
from pathlib import Path

METADATA_FILE = Path(f"/share/nas2_3/jfont/ILP-Radio-Array/metadata_heuristic_ska_full.parquet")
SOLUTIONS_FILE = Path(f"/share/nas2_3/jfont/ILP-Radio-Array/solutions_heuristic_ska_full.npz")


def append_metadata(metadata):
    df = pd.DataFrame([metadata])
    if METADATA_FILE.exists():
        existing = pd.read_parquet(METADATA_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(METADATA_FILE, index=False)


def append_solution(key, solution):
    if SOLUTIONS_FILE.exists():
        sols = dict(np.load(SOLUTIONS_FILE, allow_pickle=True))
    else:
        sols = {}
    sols[key] = np.array(solution, dtype=object)
    np.savez_compressed(SOLUTIONS_FILE, **sols)


def compute_bin_number(graph, subarray_number, edge_number_per_bin=6):
    edge_number = graph.number_of_edges() / subarray_number
    bin_number = max(2, int(np.floor(edge_number / edge_number_per_bin)))
    return bin_number


def run_optimiser(graph, subarray_number, seed):
    start = time.time()
    solution, cost = solver(graph, subarray_number, p=1, seed=seed)
    end = time.time()

    sol_key = f"node{len(list(graph.nodes()))}_sub{subarray_number}_seed{seed}"
    append_solution(sol_key, solution)

    metadata = {
        "seed": seed,
        "node_num": len(list(graph.nodes())),
        "subarray_number": subarray_number,
        "cost": cost,
        "optimisation_time": end - start,
        "solution_key": sol_key
    }
    append_metadata(metadata)

    return solution, metadata


def main():
    seeds = [137, 58291, 9021, 47717, 26539]


    all_runs = [(seed)
                for seed in seeds
                ]

    if METADATA_FILE.exists():
        done_df = pd.read_parquet(METADATA_FILE)
        done_set = set(zip(done_df.node_num, done_df.seed))
    else:
        done_set = set()

    runs_to_do = [run for run in all_runs if run not in done_set]
    total_runs = len(runs_to_do)

    for i, (seed) in enumerate(runs_to_do, start=1):
        print(f"Running {i}/{total_runs}: Seed={seed}")
        graph, _, _ = gl.ska_mid_full_graph()
        _, _ = run_optimiser(graph, 2, seed)

if __name__ == '__main__':
    main()
