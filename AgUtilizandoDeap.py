import random
import numpy as np
from deap import algorithms, base, creator, tools

# Define the problem
def load_tsp(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        node_coord_section = False
        dimension = None
        node_coords = []
        edge_weight_section = False
        distances = []
        for line in lines:
            if line.startswith('DIMENSION'):
                dimension = int(line.split()[-1])
            elif line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
            elif node_coord_section:
                _, x, y = line.split()
                node_coords.append((float(x), float(y)))
            elif line.startswith('EDGE_WEIGHT_SECTION'):
                edge_weight_section = True
                continue
            elif edge_weight_section:
                distances.extend([int(x) for x in line.split()])
        distances = np.array(distances).reshape((dimension, dimension))
        for i in range(dimension):
            for j in range(i, dimension):
                distances[j, i] = distances[i, j]
        return distances

def create_tour(size):
    return random.sample(range(size), size)

def evaluate_tour(individual, distances):
    return sum([distances[individual[i], individual[(i+1)%len(individual)]] for i in range(len(individual))]),

def mutate_tour(individual):
    i, j = sorted(random.sample(range(len(individual)), 2))
    individual[i:j] = reversed(individual[i:j])
    return individual,

# Load the ATSP problem from a TSPLIB
distances = load_tsp('data/br17.atsp')

# Create the DEAP toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("tour", create_tour, size=len(distances))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.tour)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_tour, distances=distances)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", mutate_tour)
toolbox.register("select", tools.selTournament, tournsize=4)

# Run the genetic algorithm to solve the ATSP problem
population = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.4, ngen=1000, stats=stats, halloffame=hof)

# Use the 2-opt algorithm to improve the best solution found by the genetic algorithm
best_tour = hof[0]
best_dist = evaluate_tour(best_tour, distances)[0]
for i in range(len(best_tour)-1):
    for j in range(i+1, len(best_tour)):
        new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
        new_dist = evaluate_tour(new_tour, distances)[0]
        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist

# Print the best solution found
print("Best dist:", best_dist, "Best tour:", best_tour)
