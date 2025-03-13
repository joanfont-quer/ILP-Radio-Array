from Partitioner.multipart_gurobi import solve_bins
from visualisation import *
import graph_loader as gr
import time


def main():
    graph, positions = gr.generate_graph(40, 20)

    bin_number = 5
    subarray_number = 3

    start = time.time()
    solution, cost = solve_bins(graph, bin_number, subarray_number)
    end = time.time()

    print("cost: ", cost)
    print("time: ", end - start)

    visualise_sol(solution, positions, graph)
    visualise_bins(solution, graph, bin_number, subarray_number)

if __name__ == '__main__':
    main()
