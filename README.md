# README: Structure and Flow of the One-Max Problem, Evolving to a Target String, Deceptive Landscapes problems and Bin-Packing Problem.

# Introduction
This README outlines the structure and flow of the Python scripts designed to solve the One-Max Problem, the Evolving to a Target String, the Deceptive Landscapes problems and the Bin-Packing problem using a Genetic Algorithm (GA). 
- The One-Max Problem involves finding a binary string (genome) within a population that maximises the count of 1's. (one_max_problem.py)
- The Evolving to a Target String involves creating a binary string of randomly chosen 0's and 1's and finding the number of matching characters with the target string. (evolving_to_target_string.py)
- The Deceptive Landscapes involves finding a binary string within a population that maximises the count of 1's, however the fitness is artificially inflated if no 1's are present therefore introducing a local optimum into the fitness landscape. (deceptive_landscapes.py)
- The Bin-Packing problem involves efficiently allocating objects of different volumes into a finite number of bins of fixed capacity in a way that minimises the number of bins used. (bin_packing_problem.py)

# Entry Point: (All scripts) 
Each script's execution begins with the main() function. It acts as the entry point, initiating the genetic algorithm's process and capturing its performance across generations.

# Initial Population Setup: (One-Max Problem/Evolving to a Target String/Deceptive Landscapes)
The first step involves generating an initial population of random binary strings using the initialise_population() function. Each string's length is determined by GENOME_LENGTH, and the total number of strings by POPULATION_SIZE.

# Fitness Evaluation: (One-Max Problem/Evolving to a Target String/Deceptive Landscapes)
For each genome in the population, its fitness is evaluated using the fitness() function. , aligning with the  objective.
- One-Max Problem: The fitness score is the sum of 1's in the genome.
- Evolving to a Target String: The fitness score counts the number of 1’s indicating the number of matching characters with the target string.
- Deceptive Landscapes: The fitness score counts the number of 1’s, if no 1’s are present, increase the fitness score via doubling the size of the chromosome.

# Selection Process: (All scripts)
The tournament_selection() function simulates a mini-competition among randomly chosen individuals from the population. The (3) individuals with the highest fitness scores are selected as parents for the next generation.

# Crossover Operation: (One-Max Problem/Evolving to a Target String/Deceptive Landscapes)
The crossover() function combines parts of two parent genomes to produce offspring. This process introduces variation in the population, allowing some exploring of the solution space.

# Crossover Operation: (Bin-Packing Problem)
Executes a two_point_crossover() between two parent solutions to produce two offspring. This method selects two points within the solutions and swaps the segments between these points from one parent to the other, ensuring that item groups are not split.

# Mutation Operation: (All scripts)
Post-crossover, the mutate() function introduces random mutations in the offspring genomes. It flips bits based on a mutation probability (MUTATION_RATE), keeping genetic diversity.

# Elitism Strategy: (Evolving to a Target String/Deceptive Landscapes/Bin-Packing problem)
To retain the best solutions across generations, an elitism strategy is employed. The best-performing individual from the population is directly passed to the next generation without undergoing any crossover or mutation.

# Evolution Over Generations: (All scripts)
The GA iteratively evolves the population over a predefined number (100) of generations (GENERATIONS). In each iteration, it applies selection, crossover, and mutation to generate a new population. The average fitness of the population is tracked after each generation.

# Data Recording and Visualisation: (One-Max Problem/Evolving to a Target String/Deceptive Landscapes) 
The genetic_algorithm() function records the average fitness of the population for each generation. This data is saved to the csv files:
- 'fitness_history.csv',
- 'fitness_history_target.csv',
- 'fitness_history_deceptive.csv'.
 The plotting tool Gnuplot is used to visualise this data, showing the algorithm's performance and convergence over time.

# Data Recording and Visualisation: (Bin-Packing problem)
The genetic_algorithm() function records the average fitness of the bin-packing for each generation. This data is visualised via the visualise_solution() function. This outputs the data as a bar-chart.

# Conclusion
By executing these scripts, users can observe how a GA evolves solutions, demonstrating how genetic algorithms solve optimisation problems.
