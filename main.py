from Partitioner.multipart_gurobi import solve_pdf_bins_lambda
from Partitioner.multipart_heuristic import solver
from utils import describe_similar_subarray, build_subarrays_from_assignments
import graph_loader as gl
import time
import json


def main():
    generating_start = time.time()
    graph, positions_3d, positions_2d = gl.ska_mid_graph()
    generating_end = time.time()

    print("generating time: ", generating_end - generating_start)

    bin_number = 4
    subarray_number = 2

    start = time.time()
    solution, cost = solve_pdf_bins_lambda(graph, bin_number, subarray_number, [3e6, 5e6])
    end = time.time()

    print("cost: ", cost)
    print("optimisation time: ", end - start)

    print("saving solution...")
    with open("subarray_solution.json", "w") as f:
        json.dump(solution, f)

    print("plotting solution...")
    subarrays = build_subarrays_from_assignments(solution)

    describe_similar_subarray(subarrays, "MID", 5, 2, 1, 1)

if __name__ == '__main__':
    main()
