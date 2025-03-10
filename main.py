from solver import solve_kl_divergence
from visualise import *
import graph_reader as gr
import time


def main():
    graph, positions = gr.generate_graph(40, 20)

    bin_number = 10

    start = time.time()
    solution, cost = solve_kl_divergence(graph, bin_number)
    end = time.time()

    print("cost: ", cost)
    print("time: ", end - start)

    visualise_sol(solution, positions, graph)
    visualise_bins(solution, graph, bin_number)


if __name__ == '__main__':
    main()
