#!/usr/bin/env python
from itertools import compress
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

def initial_population(individual_size, population_size):
    return np.random.choice(a=[False, True], size=(population_size, individual_size))

def fitness_old(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def fitness(items: pd.DataFrame, knapsack_max_capacity: int, population: np.ndarray) -> np.ndarray:
    total_weight = population @ items['Weight']
    total_value = population @ items['Value']
    total_value[total_weight > knapsack_max_capacity] = 0
    return total_value

def population_best_old(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def population_best(population: np.ndarray, fitnesses: np.ndarray) -> tuple[np.ndarray, int]:
    best_individual = population[np.argmax(fitnesses)]
    best_individual_fitness = np.max(fitnesses)
    return best_individual, best_individual_fitness

items, knapsack_max_capacity = get_big()
print(items)

population_size = 10000
generations = 2000
n_selection = 2000
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
fitnesses = fitness(items, knapsack_max_capacity, population)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # 1. Selection
    selected_parents_idx = np.random.choice(len(population), size=n_selection, p=fitnesses/fitnesses.sum())
    selected_parents = population[selected_parents_idx]

    # 2. Crossover
    half = len(items) // 2
    children11 = selected_parents[:n_selection-1, :half]
    children12 = selected_parents[1:n_selection, half:]
    children21 = selected_parents[1:n_selection, :half]
    children22 = selected_parents[:n_selection-1, half:]
    children = np.array(np.bmat([[children11, children12], [children21, children22]]))

    # 3. Mutation
    # flip one random bit in each child
    flip_locations = np.random.randint(len(items), size=len(children))
    children[range(len(children)), flip_locations] ^= True

    # 4. Update population
    selected_survivors_idx = np.random.choice(len(population), size=population_size-len(children), p=fitnesses/fitnesses.sum())
    selected_survivors = population[selected_survivors_idx]
    population = np.concatenate((children, selected_survivors))

    # recalculate fitnesses
    fitnesses = fitness(items, knapsack_max_capacity, population)

    best_individual, best_individual_fitness = population_best(population, fitnesses)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = fitness(items, knapsack_max_capacity, population)
    population_fitnesses[::-1].sort()
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
