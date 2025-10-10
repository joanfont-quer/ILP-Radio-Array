from Partitioner.multipart_evolutionary import GeneticAlgorithm
from Partitioner.multipart_heuristic import wasserstein
import graph_loader as gl
import visualisation as vis


graph, positions_3d, positions_2d = gl.ska_mid_full_graph()
ga = GeneticAlgorithm(graph, partition_number=2, objective=wasserstein, objective_parameters={"p": 1, "alpha": 1},
                      pop_size=100, n_gen=650, seed=42)

ga.solve()
ga.save_history("gen_history.pkl")

x, f = ga.best_solution()
solution_dict = {node: part for node, part in zip(ga.problem.nodes, x)}

vis.visualise_sol(solution_dict, positions_2d, graph)
vis.visualise_bins(solution_dict, graph, 50)
