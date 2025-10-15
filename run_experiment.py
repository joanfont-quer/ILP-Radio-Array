import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import time
from filelock import FileLock
from utils import load_config
import graph_loader as gl

BASE_PATH = Path(__file__).resolve().parent


def safe_json(val):
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    elif isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, Path):
        return str(val)
    elif callable(val):
        return getattr(val, "__name__", str(val))
    elif isinstance(val, dict):
        return {k: safe_json(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [safe_json(x) for x in val]
    else:
        try:
            json.dumps(val)
            return val
        except Exception:
            return str(val)


def extract_config_dict(config):

    return {
        k: safe_json(v)
        for k, v in config.__dict__.items()
        if k.isupper()
    }


def append_metadata(metadata: dict, metadata_file: Path):
    df = pd.DataFrame([metadata])
    lock = FileLock(str(metadata_file) + ".lock")
    with lock:
        if metadata_file.exists():
            existing = pd.read_parquet(metadata_file)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(metadata_file, index=False)


def append_solution(key, solution, solution_file: Path):
    lock = FileLock(str(solution_file) + ".lock")
    with lock:
        if solution_file.exists():
            sols = dict(np.load(solution_file, allow_pickle=True))
        else:
            sols = {}
        sols[key] = np.array(solution, dtype=object)
        np.savez_compressed(solution_file, **sols)


def run(config, graph_func, node_nums, seeds, subarray_nums=2):
    exp_dir = BASE_PATH / graph_func.__name__ / config.NAME

    exp_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = exp_dir / "metadata.parquet"
    solution_file = exp_dir / "solution.npz"

    config_dict = extract_config_dict(config)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    existing_metadata = pd.read_parquet(metadata_file) if metadata_file.exists() else pd.DataFrame()
    if not existing_metadata.empty and "solution_key" in existing_metadata.columns:
        existing_keys = set(existing_metadata["solution_key"])
    else:
        existing_keys = set()

    node_nums = node_nums if isinstance(node_nums, (list, tuple)) else [node_nums]

    subarray_nums = subarray_nums if isinstance(subarray_nums, (list, tuple)) else [subarray_nums]

    if seeds is not None:
        seeds = seeds if isinstance(seeds, (list, tuple)) else [seeds]
    else:
        seeds = [137, 58291, 9021, 47717, 26539]

    for node_num in node_nums:
        for seed in seeds:
            for subarray_num in subarray_nums:
                key = f"node{node_num}_sub{subarray_num}_seed{seed}"
                if key in existing_keys:
                    continue

                print(f"Running node={node_num}, sub={subarray_num}, seed={seed}")

                graph, *rest = graph_func(node_num=node_num, seed=seed)

                start = time.time()
                try:
                    solver = config.solver(graph=graph, seed=seed, subarray_num=subarray_num)
                    solution, cost = solver.solve()
                except Exception as e:
                    print(f"Error for {key}: {e}")
                    continue
                end = time.time()

                metadata = dict(
                    solution_key=key,
                    node_num=node_num,
                    subarray_num=subarray_num,
                    seed=seed,
                    cost=cost,
                    optimisation_time=end-start,
                )

                append_metadata(metadata, metadata_file)
                append_solution(key, solution, solution_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration .py file")
    parser.add_argument("--graph_func", required=True, help="Graph function")
    parser.add_argument("--node_nums", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds")
    parser.add_argument("--subarray_nums", nargs="+", type=int, help="Subarray numbers", default=[2])

    args = parser.parse_args()

    config = load_config(args.config)

    graph_func = getattr(gl, args.graph_func)

    run(config, graph_func, args.node_nums, args.seeds, args.subarray_nums)


if __name__ == "__main__":
    main()