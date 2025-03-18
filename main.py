from Partitioner.multipart_gurobi import solve_kl
from visualisation import *
import graph_loader as gr
import time


def main():
    graph, positions = gr.read_tsp_file("Problems/berlin52.tsp")

    bin_number = 10
    subarray_number = 2

    start = time.time()
    solution, cost = solve_kl(graph, subarray_number)
    end = time.time()

    print("cost: ", cost)
    print("time: ", end - start)

    visualise_sol(solution, positions, graph)
    visualise_bins(solution, graph, bin_number, subarray_number)

if __name__ == '__main__':
    main()
