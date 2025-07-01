from deap import base, creator, tools, algorithms
import numpy as np
import random


class TSP_GA_Optimizer:
    def __init__(self, distance_matrix, depot_index=0):
        self.distance_matrix = distance_matrix
        self.depot_index = depot_index
        self.num_locations = distance_matrix.shape[0]
        self._setup_ga()

    def _setup_ga(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Safer individual creation
        self.toolbox.register("indices", random.sample,
                              range(1, self.num_locations),  # Skip depot
                              self.num_locations - 1)  # Number of deliveries

        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              self._create_valid_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Modified genetic operators with validation
        self.toolbox.register("mate", self._safe_cxOrdered)
        self.toolbox.register("mutate", self._safe_mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate)

    def _create_valid_individual(self):
        """Ensure we always create valid individuals"""
        while True:
            try:
                ind = self.toolbox.indices()
                if len(set(ind)) == len(ind):  # Check for duplicates
                    return ind
            except:
                continue

    def _safe_cxOrdered(self, ind1, ind2):
        """Wrapper for ordered crossover that preserves individual type"""
        try:
            # Create proper DEAP individuals
            ind1_copy = creator.Individual(ind1.copy())
            ind2_copy = creator.Individual(ind2.copy())

            # Perform crossover
            tools.cxOrdered(ind1_copy, ind2_copy)

            # Validate results
            if len(set(ind1_copy)) == len(ind1_copy) and len(set(ind2_copy)) == len(ind2_copy):
                return ind1_copy, ind2_copy
            return ind1, ind2
        except:
            return ind1, ind2

    def _safe_mutShuffleIndexes(self, individual, indpb):
        """Wrapper for mutation that preserves individual type"""
        try:
            # Create proper DEAP individual
            mut_ind = creator.Individual(individual.copy())

            # Perform mutation
            tools.mutShuffleIndexes(mut_ind, indpb)

            # Validate
            if len(set(mut_ind)) == len(mut_ind):
                return (mut_ind,)
            return (individual,)
        except:
            return (individual,)

    def _evaluate(self, individual):
        """Evaluation with validation"""
        try:
            # Check individual validity first
            if len(individual) != len(set(individual)):
                return (float('inf'),)

            total_distance = 0
            total_distance += self.distance_matrix[self.depot_index][individual[0]]

            for i in range(len(individual) - 1):
                total_distance += self.distance_matrix[individual[i]][individual[i + 1]]

            total_distance += self.distance_matrix[individual[-1]][self.depot_index]
            return (total_distance,)
        except:
            return (float('inf'),)  # Return worst possible fitness if error

    def optimize(self, population_size=100, generations=200, cx_prob=0.9, mut_prob=0.1):
        """Run optimization with additional validation"""
        pop = self.toolbox.population(n=population_size)

        # Add periodic validation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("valid", lambda pop: sum(1 for ind in pop if len(ind) == len(set(ind))) / len(pop))

        hof = tools.HallOfFame(1)

        try:
            pop, log = algorithms.eaSimple(
                pop, self.toolbox, cxpb=cx_prob, mutpb=mut_prob,
                ngen=generations, stats=stats, halloffame=hof, verbose=True
            )
        except Exception as e:
            print(f"Optimization interrupted: {str(e)}")
            print("Returning best individual found so far")

        best_route = [self.depot_index] + hof[0] + [self.depot_index]
        best_distance = self._evaluate(hof[0])[0]

        return {
            'best_route': best_route,
            'best_distance': best_distance,
            'log': log if 'log' in locals() else None
        }