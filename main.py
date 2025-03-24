from Partitioner.multipart_gurobi import solve_kl, solve_bins
from Partitioner.multipart_heuristic import solver
from visualisation import *
import graph_loader as gl
import time


def main():
    graph, positions = gl.generate_graph_gaussian(100, 20)

    bin_number = 20
    subarray_number = 3

    start = time.time()
    solution, cost = solver(graph, subarray_number, 1/2)
    end = time.time()

    print("cost: ", cost)
    print("time: ", end - start)

    visualise_sol(solution, positions, graph)
    visualise_bins(solution, graph, bin_number, subarray_number)

if __name__ == '__main__':
    main()
