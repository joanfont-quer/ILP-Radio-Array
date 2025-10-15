import random
import bisect


class HeuristicGraphPartitioner:
    def __init__(self, graph, partition_number, objective, objective_parameters, seed=42):
        """
        Graph partitioning solver that uses greedy assignment with a user-defined objective function.
        Args:
            graph (networkx.Graph): The input weighted graph.
            partition_number (int): Number of partitions to divide the graph into.
            objective (callable): Objective function to evaluate partition quality (to be minimised).
            objective_parameters (dict): Dictionary containing objective parameters.
            seed (int): Random seed used for reproducibility.
        """
        self.graph = graph
        self.unassigned = list(graph.nodes())
        self.y = {node: -1 for node in graph.nodes()}
        self.partition_number = partition_number
        self.partition_weights = {g: [] for g in range(partition_number)}
        self.objective = objective
        self.objective_parameters = objective_parameters
        self.seed = seed
        random.seed(seed)

    def assign_node(self):
        """
        Assigns one unassigned node to a partition while minimising the overall Wasserstein cost.

        The method evaluates each candidate assignment of every unassigned node to every partition. For each trial,
        edge weights are temporarily added to the target partition, the objective is computed, and the insertion is
        rolled back. After exploring all possible options, the best assignment is commited to permanently.
        """
        best_dist = float('inf')
        best_assignment = (None, None, None)

        for node in self.unassigned:
            for partition in range(self.partition_number):
                added_weights = []
                self.y[node] = partition

                for neighbor in self.graph.neighbors(node):
                    if self.y[neighbor] == partition:
                        w = self.graph[node][neighbor]['weight']
                        bisect.insort(self.partition_weights[partition], w)
                        added_weights.append(w)

                current_dist = self.objective(self.partition_weights, self.objective_parameters)
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_assignment = (node, partition, added_weights.copy())

                # Roll back weights to before assignment trial
                for w in added_weights:
                    self.partition_weights[partition].remove(w)
                self.y[node] = -1

        node, partition, added_weights = best_assignment
        self.y[node] = partition
        self.unassigned.remove(node)
        for w in added_weights:
            bisect.insort(self.partition_weights[partition], w)


    def solve(self):
        """
        Executes the greedy partitioning algorithm until all nodes are assigned.

        The algorithm begins by randomly assigning one node to each partition. Then, while unassigned nodes remain, it
        repeatedly assigns a node to a partition such that it minimises the selected objective.

        Returns:
            tuple : (y, cost)
            y (dict): Dictionary mapping nodes to their assigned partitions.
            cost (float): Final value of the objective function after all assignments.
        """
        for partition in range(self.partition_number):
            node = random.choice(self.unassigned)
            self.y[node] = partition
            self.unassigned.remove(node)

        while len(self.unassigned) > 0:
            self.assign_node()

        cost = self.objective(self.partition_weights, self.objective_parameters)
        return self.y, cost
