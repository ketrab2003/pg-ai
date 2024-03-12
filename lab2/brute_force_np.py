from itertools import product, compress, chain
import time
import numpy as np

from data import *

def fitness(items: pd.DataFrame, knapsack_max_capacity: int, population: np.ndarray) -> np.ndarray:
    total_weight = population @ items['Weight']
    total_value = population @ items['Value']
    total_value[total_weight > knapsack_max_capacity] = 0
    return total_value

def single_fitness(items: pd.DataFrame, knapsack_max_capacity: int, solution: np.ndarray):
    weight = items['Weight'] @ solution
    value = items['Value'] @ solution
    if weight > knapsack_max_capacity:
        return 0
    return value

items, knapsack_max_capacity = get_big()
print(items)

block_items_count = 15

start_time = time.time()
best_solution = None
best_value = 0

block_population = np.fromiter(chain(*product([False, True], repeat=block_items_count)), dtype=bool).reshape(-1, block_items_count)
values = block_population @ items[:block_items_count]['Value']
weights = block_population @ items[:block_items_count]['Weight']

for solution in product([False, True], repeat=len(items)-block_items_count):
    solution_value = sum(compress(items[block_items_count:]['Value'], solution))
    solution_weight = sum(compress(items[block_items_count:]['Weight'], solution))
    values += solution_value
    weights += solution_weight

    solution_results = values[weights <= knapsack_max_capacity]
    if len(solution_results) == 0:
        solution_best = 0
    else:
        solution_best = solution_results.max()

    if solution_best > best_value:
        best_from_block = np.argwhere(values == solution_best).flatten()[0]
        best_solution = tuple(block_population[best_from_block]) + solution
        best_value = solution_best
    
    values -= solution_value
    weights -= solution_weight

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_value)
print('Time: ', total_time)