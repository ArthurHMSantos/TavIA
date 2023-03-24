import random
import numpy as np


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


# Create a random tour
def create_tour(size):
    return random.sample(range(size), size) # Create a random tour of the correct size


# Calculate the fitness of an individual
def calculate_fitness(individual, distances): 
    fitness = 0
    for i in range(len(individual)): # Loop through the nodes in the individual
        j = (i + 1) % len(individual) # Get the index of the next node in the circular route
        city_i = individual[i]
        city_j = individual[j]
        fitness += distances[city_i, city_j]
    return fitness


# Create the initial population
def generate_population(pop_size, distances):
    population = []
    while len(population) < pop_size:
        individual = create_tour(distances.shape[0]) # Create a random tour of the correct size
        if individual not in population: # Check if the individual is already in the population
            population.append(individual)
    # Calculate the fitness of each individual in the population
    fitness_values = [calculate_fitness(individual, distances) for individual in population]
    return population, fitness_values


# Calculate the fitness of the entire population
def calculate_fitness_pop(population, distances):
    fitness_values = [calculate_fitness(individual, distances) for individual in population]
    return fitness_values


# Select the best individuals from the population with rank selection
def rank_selection(population, fitness_values, num_parents):
    ranked_population = [individual for _, individual in sorted(zip(fitness_values, population))] # Sort the population based on the fitness values
    selection_probs = np.linspace(1, 0, len(ranked_population)) / sum(np.linspace(1, 0, len(ranked_population))) # Calculate the selection probabilities
    selected_parents = random.choices(ranked_population, weights=selection_probs, k=num_parents) # Select the parents
    return selected_parents


# Perform the crossover operation - in this case we using the order crossover
def ox_crossover(parent1, parent2):
    # Choose two random indices for the slice
    idx1, idx2 = sorted(random.sample(range(len(parent1)), 2)) # Sort the indices to ensure idx1 < idx2
    # Initialize the child with the same elements as parent 2
    child = [-1] * len(parent2)
    # Copy the slice from parent 1 to child
    child[idx1:idx2] = parent1[idx1:idx2]
    # Find the elements in parent 2 that are not in the slice
    remaining = [gene for gene in parent2 if gene not in child[idx1:idx2]]
    # Fill in the remaining elements in the child
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining.pop(0)
    return child


# Calls the crossover operation and calculates the fitness of the child
def crossover(parents, distances):
    parent1, parent2 = parents
    child = ox_crossover(parent1, parent2)
    return child, calculate_fitness(child, distances)


# Generate the offspring from the parents
def generate_offspring(parents, distances, crossover_rate, mutation_rate):
    if random.random() < crossover_rate:
        child, child_fitness = crossover(parents, distances)
    else:
        child = random.choice(parents)
        child_fitness = calculate_fitness(child, distances)
        
    if random.random() < mutation_rate:
        child = em_mutation(child)
        child_fitness = calculate_fitness(child, distances)
        
    return child, child_fitness


# Generate the offspring from the parents population
def generate_offspring_population(parents, distances, crossover_rate, mutation_rate):
    children = []
    while len(children) < len(parents):
        parent1, parent2 = random.sample(parents, 2)
        child, child_fitness = generate_offspring((parent1, parent2), distances, crossover_rate, mutation_rate)
        children.append(child)
    return children


# Perform the mutation operation - in this case we using the swap mutation
def em_mutation(individual):
    # Escolhe duas cidades aleatórias
    city1, city2 = random.sample(range(len(individual)), 2)
    # Troca as duas cidades de lugar na rota
    individual[city1], individual[city2] = individual[city2], individual[city1]
    return individual


# Generate the offspring from the parents population
# Genetic algorithm
def genetic_algorithm(distances, pop_size, num_generations, crossover_rate, mutation_rate):
    # Create the initial population
    population, fitness_values = generate_population(pop_size, distances)
    
    # Loop through each generation
    for i in range(num_generations):
        # Select the parents using rank selection
        parents = rank_selection(population, fitness_values, 200)
        
        # Generate the offspring from the parents
        children = generate_offspring_population(parents, distances, crossover_rate, mutation_rate)
        
        # Replace the parents with the children
        population = children
        fitness_values = calculate_fitness_pop(population, distances)
        
        # Print the best individual in the current generation
        best_fitness_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_fitness_idx]
        best_individual = population[best_fitness_idx]
        print(f"Generation {i+1}: Best fitness = {best_fitness}, Best individual = {best_individual}")
    
    # Return the best individual in the final generation
    best_fitness_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[best_fitness_idx]
    best_individual = population[best_fitness_idx]
    return best_individual, best_fitness


if __name__ == '__main__':
    # Set the random seed
    #random.seed(42)
    # Load the TSP problem from a TSPLIB
    distances = load_tsp('data/swiss42.tsp')
    
    # Run the genetic algorithm with 1000 generations and a population size of 100 individuals  
    best_individual, best_fitness = genetic_algorithm(distances, pop_size=200, num_generations=10000, crossover_rate=0.8, mutation_rate=0.1)
    
    # Print the best individual and its fitness value
    print(f"Best individual: {best_individual}")
    print(f"Best fitness value: {best_fitness}")