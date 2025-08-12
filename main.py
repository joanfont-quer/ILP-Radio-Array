from Partitioner.multipart_gurobi import solve_pdf_bins
import graph_loader as gl
import numpy as np
import pandas as pd
import time
from pathlib import Path

METADATA_FILE = Path("metadata.parquet")
SOLUTIONS_FILE = Path("solutions.npz")


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


def run_optimiser(graph, subarray_number, node_number, seed):
    start = time.time()
    bin_number = compute_bin_number(graph, subarray_number)
    solution, cost = solve_pdf_bins(graph, bin_number, subarray_number)
    end = time.time()

    sol_key = f"node{node_number}_sub{subarray_number}_seed{seed}"
    append_solution(sol_key, solution)

    metadata = {
        "seed": seed,
        "node_num": node_number,
        "subarray_number": subarray_number,
        "bin_number": bin_number,
        "cost": cost,
        "optimisation_time": end - start,
        "solution_key": sol_key
    }
    append_metadata(metadata)

    return solution, metadata


def main():
    seeds = [137, 58291, 9021, 47717, 26539]

    all_runs = [(node_number, subarray_number, seed)
                for node_number in range(4, 83)
                for subarray_number in range(2, 5)
                for seed in seeds
                if 2 * subarray_number <= node_number
                ]

    if METADATA_FILE.exists():
        done_df = pd.read_parquet(METADATA_FILE)
        done_set = set(zip(done_df.node_num, done_df.subarray_number, done_df.seed))
    else:
        done_set = set()

    runs_to_do = [run for run in all_runs if run not in done_set]
    total_runs = len(runs_to_do)

    for i, (node_number, subarray_number, seed) in enumerate(runs_to_do, start=1):
        print(f"Running {i}/{total_runs}: node={node_number}, sub={subarray_number}, seed={seed}")
        graph, _ = gl.generate_graph(node_number, seed)
        _, _ = run_optimiser(graph, subarray_number, node_number, seed)

if __name__ == '__main__':
    main()
