import random

# Parameters for genetic algorithm.
POPULATION_SIZE = 100  # Size of the population.
GENOME_LENGTH = 30     # Length of each binary string (genome).
GENERATIONS = 100      # Number of generations to evolve.
MUTATION_RATE = 0.01   # Probability of each bit being mutated.
CROSSOVER_RATE = 0.8  # Probability of crossover occurring.
ELITISM = True         # Enabled elitism so the best solution found is not lost due to crossover or mutation.
TARGET_STRING = [random.randint(0,1) for _ in range(GENOME_LENGTH)]  # Random target string of 0's and 1's.


# Function to initialise the population with random binary strings.
def initialise_population():
    return [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(POPULATION_SIZE)]


# Function to calculate the fitness by comparison with the target string.
def fitness(genome):
    """Calculates the fitness of a genome based on its similarity to a predefined target string.
    The fitness is the total number of positions where the genome and the target string match.

    :param genome:  A list representing the genome, where each element is a character.

    :return: The fitness score (int) indicating the number of matching characters with the TARGET_STRING.
    """
    return sum(1 for a, b in zip(genome, TARGET_STRING) if a == b)


def tournament_selection(population, tournament_size=3):
    """Selects two parents from the population using tournament selection.

    :param population: The population from which to select parents.
    :param tournament_size: The number of individuals to compete in each tournament.

    :return winners[0], winners[1]: tuple of two selected parents.
    """
    winners = []
    for _ in range(2):  # Run two tournaments to select two parents
        competitors = random.sample(population, tournament_size)
        winner = max(competitors, key=fitness)
        winners.append(winner)
    return winners[0], winners[1]


# Function to perform crossover (exchange segments) between two parents
def crossover(parent1, parent2):
    """Perform crossover between two parent genomes to produce offspring.
    Crossover occurs with a predefined probability (CROSSOVER_RATE).
    A single point on the genome is selected at random, and the segments beyond that point are swapped between the two parents.

    :param parent1: The first parent genome.
    :param parent2: The second parent genome.

    :return: A tuple containing two offspring genomes.
    """
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENOME_LENGTH - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    else:
        return parent1, parent2


# Function to mutate a genome (flip bits with a certain probability).
def mutate(genome):
    """Mutate a genome by flipping bits with a predefined probability (MUTATION_RATE).
    Each bit in the genome has a chance of being flipped (0 to 1 or 1 to 0).

    :param genome: The genome to mutate.

    :return: The mutated genome as a list of bits.
    """
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in genome]


# Main function implementing the genetic algorithm.
def genetic_algorithm():
    """Main function implementing the genetic algorithm.
    Initialises a population, then evolves it over a number of generations (GENERATIONS),
    applying selection, crossover, and mutation to create new generations. Elitism can be applied to
    keep the best individual without mutation.

    :return: A list containing the average fitness of the population for each generation.
    """
    population = initialise_population()  # Initialise the population.
    average_fitness_history = []          # List to track average fitness over generations.

    for _ in range(GENERATIONS):
        next_population = []    # List to populate the next population

        # Implementing Elitism: Keeping the best performing individual.
        if ELITISM:
            best_individual = max(population, key=fitness)
            next_population.append(best_individual)

        # Fill with rest of the next population.
        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = tournament_selection(population)  # Select two parents.
            offspring1, offspring2 = crossover(parent1, parent2)  # Create offspring via crossover.
            next_population.extend([mutate(offspring1), mutate(offspring2)])  # Mutate offspring and add to next population.

        population = next_population[:POPULATION_SIZE]  # Ensuring population size remains constant

        # Calculate and store the average fitness of the current generation.
        average_fitness = sum(fitness(genome) for genome in population) / POPULATION_SIZE
        average_fitness_history.append(average_fitness)

    return average_fitness_history


def main():
    # Run the genetic algorithm and print the average fitness for each generation
    average_fitness_history = genetic_algorithm()

    # Writing the average fitness values to a file
    with open('fitness_history_target.csv', 'w') as file:
        for generation, avg_fitness in enumerate(average_fitness_history):
            file.write(f"{generation},{avg_fitness}\n")
            print(f"Generation {generation}: Average Fitness = {avg_fitness}")


if __name__ == "__main__":
    main()
