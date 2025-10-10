import pickle as pkl
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import bisect
import copy


class UniformIntegerMutation(Mutation):
    """
    Uniform integer mutation operator.

    Each decision variable in each individual is mutated with equal probability. A mutated variable is replaced with
    a new integer drawn from a uniform distribution of all allowed integer values excluding the current value of the
    variable.

    Args:
        prob (float): Overall mutation probability per individual.
        at_least_once (bool): If True, ensures at least one variable per individual is mutated.
            Not currently implemented.
        **kwargs: Additional arguments passed to the base Mutation class.

    """
    def __init__(self, prob=0.1, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once

    def _do(self, problem, X, **kwargs):
        """
        Apply uniform integer mutation operator to a population.

        Args:
            problem (Problem): Optimisation problem with integer bounds and variables.
            X (ndarray): Population to mutate.
            **kwargs:

        Returns:
            Xp (ndarray): Mutated population.
        """
        Xp = X.astype(int, copy=True)
        n_individuals, n_var = Xp.shape

        xl, xu = problem.xl.astype(int), problem.xu.astype(int)

        prob_var = self.get_prob_var(problem, size=n_individuals)

        mut_mask = np.random.rand(n_individuals, n_var) < prob_var[:, None]

        # If no variables selected for mutation, skip the rest of the computation.
        if not np.any(mut_mask):
            return Xp

        range_size = xu[0] - xl[0] + 1
        offsets = np.random.randint(1, range_size, size=Xp.shape)
        # Add the offset to the mutated variables with modulo addition.
        rand_ints = xl[0] + ((Xp - xl[0] + offsets) % range_size)

        Xp[mut_mask] = rand_ints[mut_mask]
        return Xp


class UniformIntegerCrossover(Crossover):
    """Uniform crossover for integer decision variable problems.

    Each pair of parents produces one or two offspring by exchanging decision variable independently with a given
    probability of the variable coming from the 1st or 2nd parent.
    Args:
        prob_var (float, optional): Probability of variable coming from 1st parent, defaults to 0.5.
        n_offspring (int, optional): Number of offspring generated per mating. Defaults to 2.
        prob_exchange (float, optional): Probability that crossover occurs for a given mating pair, defaults to 1.0.
        **kwargs: Additional arguments passed to the base Crossover class.
    """
    def __init__(self, prob_var=0.5, n_offsprings=2, prob_exchange=1.0, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.prob_var = prob_var
        self.prob_exchange = prob_exchange

    def _do(self, problem, X, **kwargs):
        """
        Perform uniform integer crossover on mating pairs.

        Args:
            problem (Problem): Optimisation problem (unused, necessary for consistency with other crossover operators).
            X (ndarray): Array of shape (2, n_matings, n_var) containing the parent populations.

        Returns:
            Q (ndarray): Offspring array of shape (n_offsprings, n_matings, n_var).
        """
        _, n_matings, n_var = X.shape
        p1, p2 = X[0], X[1]

        do_crossover = np.random.rand(n_matings) < self.prob_exchange

        c1, c2 = p1.copy(), p2.copy()

        # Select the value of each variable from both parents, the 2nd offspring is the complement of the 1st child.
        if np.any(do_crossover):
            mask = np.random.rand(n_matings, n_var) < self.prob_var
            mask[~do_crossover, :] = False

            tmp = c1.copy()
            c1[mask] = c2[mask]
            c2[mask] = tmp[mask]

        if self.n_offsprings == 1:
            Q = np.expand_dims(c1, 0)
        else:
            Q = np.stack([c1, c2], axis=0)
        return Q



class CheckpointCallback:
    def __init__(self, every_n_gen=5, filename="ga_algorithm.pkl"):
        self.every_n_gen = every_n_gen
        self.filename = filename
        self.gen_counter = 0

    def __call__(self, algorithm):
        self.gen_counter += 1
        if self.gen_counter % self.every_n_gen == 0:
            with open(self.filename, "wb") as f:
                pkl.dump(algorithm, f)
            print(f"Checkpoint saved at generation {self.gen_counter}")


class PartitionerProblem(ElementwiseProblem):
    def __init__(self, graph, partition_number, objective, objective_parameters):
        self.graph = graph
        self.partition_number = partition_number
        self.objective = objective
        self.objective_parameters = objective_parameters

        self.nodes = list(graph.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        n_nodes = len(self.nodes)

        super().__init__(n_var=n_nodes,
                         n_obj=1,                   # single objective
                         n_ieq_constr=self.partition_number,
                         n_eq_constr=0,
                         xl=0,                      # lower bound (partition index min)
                         xu=partition_number - 1,   # upper bound (partition index max)
                         type_var=int)              # discrete variables


    def _evaluate(self, x, out, *args, **kwargs):
        partition_weights = {p: [] for p in range(self.partition_number)}
        counts = np.zeros(self.partition_number, dtype=int)

        for u, v, data in self.graph.edges(data=True):
            p_u = int(x[self.node_to_idx[u]])
            p_v = int(x[self.node_to_idx[v]])
            if p_u == p_v:
                bisect.insort(partition_weights[p_u], data['weight'])

        for i, p in enumerate(x):
            counts[int(p)] += 1

        cost = self.objective(partition_weights, self.objective_parameters)

        out["F"] = cost

        g = np.array([2 - c for c in counts], dtype=float)
        out["G"] = g


class GeneticAlgorithm:
    def __init__(self, graph, partition_number, objective, objective_parameters=None,
                 pop_size=100, n_gen=200, seed=42):
        self.graph = graph
        self.partition_number = partition_number
        self.objective = objective
        self.objective_parameters = objective_parameters if objective_parameters is not None else {}
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed

        self.callback = CheckpointCallback(every_n_gen=5, filename="ga_algorithm.pkl")

        self.problem = PartitionerProblem(graph, partition_number, objective, objective_parameters)

        self.algorithm = GA(pop_size=self.pop_size,
                            sampling=IntegerRandomSampling(),
                            crossover=UniformIntegerCrossover(),
                            mutation=UniformIntegerMutation(prob=0.1)
                            )

        self.termination = get_termination("n_gen", self.n_gen)
        self.result = None

    def solve(self, verbose=True):
        self.result = minimize(self.problem,
                               self.algorithm,
                               self.termination,
                               seed=self.seed,
                               save_history=True,
                               callback=self.callback,
                               verbose=verbose)

        with open("ga_result.pkl", "wb") as f:
            pkl.dump(self.result, f)

    def best_solution(self):
        return self.result.X, self.result.F

    def save_history(self, file_path):
        if self.result is None or not hasattr(self.result, "history"):
            raise RuntimeError("No history available.")

        hist = copy.deepcopy(self.result.history)
        with open(file_path, "wb") as f:
            pkl.dump(hist, f)

    @staticmethod
    def load_history(filepath):
        with open(filepath, "rb") as f:
            return pkl.load(f)

    def resume_from_last(self, n_current_gen, n_extra_gen):
        with open("ga_result.pkl", "rb") as f:
            result = pkl.load(f)

        self.algorithm = result.algorithm
        self.result = minimize(self.problem,
                               self.algorithm,
                               get_termination("n_gen", n_extra_gen+n_current_gen),
                               save_history=True,
                               callback=self.callback,
                               verbose=True)
