import random
import numpy as np
from deap import algorithms, base, creator, tools

random.seed(22)

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
    # Initialize the sum of distances travelled
    total_distance = 0
    # Loop through the nodes in the individual
    for node_index in range(len(individual)):
        # Get the index of the next node in the circular route
        next_node_index = (node_index + 1) % len(individual)
        # Get the distance between the current node and the next node in the route
        distance = distances[individual[node_index], individual[next_node_index]]
        # Add the distance to the total sum
        total_distance += distance
    # Return the total sum of distances as the fitness of the individual
    return total_distance,


def mutate_tour(individual):
    i, j = sorted(random.sample(range(len(individual)), 2))
    individual[i:j] = reversed(individual[i:j])
    return individual,

# Load the ATSP problem from a TSPLIB
distances = load_tsp('data/rbg403.atsp')

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
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.4, ngen=50, stats=stats)

# Print the best solution found
best_solution = tools.selBest(population, k=1)[0]
print(f'Best solution fitness: {best_solution.fitness.values[0]}')
