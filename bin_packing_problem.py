import random
import os
import matplotlib.pyplot as plt


def read_instances(filename):
    """Reads problem instances from the file and returns a list of instances.
    Each instance is a dictionary with the bin capacity and a list of item weights and counts.

    :param filename: Input txt file.
    :return: problem_instances: List of instances stored as dictionary.
    """
    problem_instances = []  # Empty list to hold all problem instances.

    with open(filename, 'r') as file:
        lines = file.readlines()  # Read all lines from the file into a list.
        i = 0
        while i < len(lines):
            if 'BPP' in lines[i]:
                # Read the number of different item weights and bin capacity.
                num_items = int(lines[i + 1].strip())
                bin_capacity = int(lines[i + 2].strip())

                items = []  # Initialise empty list to hold items (weight, count) for this instance.
                for j in range(num_items):
                    line = lines[i + 3 + j].strip().split()  # Split each line into weight and count.
                    weight, count = int(line[0]), int(line[1])  # Convert the weight and count to integers.
                    items.append((weight, count))

                # Add the instance to the list of problem instances.
                problem_instances.append({'bin_capacity': bin_capacity, 'items': items})
                i += 3 + num_items  # Moving counter past the current problem instance.
            else:
                i += 1  # 'BPP' not in the line move to next line.

    return problem_instances


def initialise_population(problem_instance, population_size):
    """
    Initialises a population for the bin-packing problem. Each item is assigned to bins in a
    sequential manner.

    :param problem_instance: A dictionary with 'bin_capacity' indicating the capacity of each bin,
                             and 'items', a list of tuples where each tuple contains an item's weight and count.
    :param population_size: The number of solutions to generate for the population.
    :return: A list of generated solutions, where each solution is a list of bin assignments for all items.
    """
    items = problem_instance['items']
    bin_capacity = problem_instance['bin_capacity']

    # Function to find a bin for an item or create a new one.
    def find_or_create_bin(bins, item_weight):
        for i, bin_total in enumerate(bins):
            if bin_total + item_weight <= bin_capacity:
                return i  # Return existing bin index.
        bins.append(0)  # Create a new bin.
        return len(bins) - 1  # Return new bin index.

    population = []
    for _ in range(population_size):
        bins = []  # Total weight in each bin.
        solution = []

        for weight, count in items:
            for _ in range(count):
                bin_index = find_or_create_bin(bins, weight)
                bins[bin_index] += weight  # Update the bin total weight.
                solution.append(bin_index + 1)  # Store bin assignment (1-indexed).

        population.append(solution)

    return population


def fitness(solution, problem_instance):
    """Calculates the fitness of a given solution for the bin-packing problem.
    The fitness is based on penalties for overfilling and under-filling bins, as well as the total number of bins used.
    A lower fitness value indicates a better solution.

    :param solution: (list): A list representing the bin number assigned to each item.
    :param problem_instance: (dict): Contains the 'bin_capacity' and the list of 'items' (weights and counts).
    :return: int: The calculated fitness of the solution.
    """
    # Calculate the total weight of items in each bin.
    bins = calculate_bin_loads(solution, problem_instance)
    # Retrieve the capacity of bins from the problem instance.
    bin_capacity = problem_instance['bin_capacity']

    # Penalties for overfilling and under-filling.
    over_capacity_penalty = 0
    under_capacity_penalty = 0
    # Iterate through the load of each bin to calculate penalties.
    for load in bins.values():
        if load > bin_capacity:
            over_capacity_penalty += (load - bin_capacity) * 5  # Add penalty for exceeding bin capacity.
        else:
            under_capacity_penalty += (bin_capacity - load)  # Add penalty for wasting bin capacity.

    # Count the number of bins used.
    used_bins = len(bins)
    # Fitness is the sum of penalties and the number of bins used.
    # A lower value indicates a more efficient solution.
    return over_capacity_penalty + under_capacity_penalty + used_bins


def calculate_bin_loads(solution, problem_instance):
    """Calculates the total weight of items in each bin based on a given solution.

    :param solution: (list): A list indicating the bin number each item is assigned to. The position
    in the list is the item. The value at each position is the bin number.

    :param problem_instance: (dict): A dictionary detailing the problem instance. Includes
    'bin_capacity' and a list of 'items'. Each item is represented by a tuple of weight and count.

    :return bin_loads: (dict): A dictionary where keys are bin numbers and values are the total weight of items in each bin.
    """
    bin_loads = {}  # Dictionary to track the total weight in each bin.
    item_index = 0  # Counter to track the current item's position in the solution list.

    for weight, count in problem_instance['items']:
        for _ in range(count):
            bin_number = solution[item_index]
            # Update the total weight for the bin. If the bin is not in the dictionary, start with 0 and add the item's weight.
            bin_loads[bin_number] = bin_loads.get(bin_number, 0) + weight
            item_index += 1

    return bin_loads


def analyse_solution(solution, problem_instance):
    """Analyses the given solution to the bin-packing problem, providing insights into
    the number of bins used, the average weight per bin, and the distribution of weight across bins.

    :param solution: (list): A list indicating the bin number each item is assigned to.
    :param problem_instance: (dict): A dictionary containing the problem instance details,
    including item weights and their counts.
    """
    # Calculate the total weight of items in each bin based on the given solution.
    bin_loads = calculate_bin_loads(solution, problem_instance)
    # Determine the number of bins actually used in the solution.
    num_bins_used = len(bin_loads)

    # Calculate the total weight of all items from the problem instance.
    total_weight = sum(weight * count for weight, count in problem_instance['items'])

    print(f"Number of bins used: {num_bins_used}")
    print(f"Average weight per bin: {total_weight / num_bins_used:.2f}")
    print("Distribution of weight across bins:")

    for bin_num, load in sorted(bin_loads.items()):
        print(f"  Bin {bin_num}: {load} weight units")


def tournament_selection(population, t_fitness, tournament_size=3):
    """Executes a tournament selection process on a given population to select parents for the next generation.
    Tournament selection involves randomly picking a set number of individuals from the population and then selecting
    the fittest to be a parent. This process is repeated until the required number of parents is selected.

    :param population: (list): A list of all current solutions in the population.
    :param t_fitness: (list): A parallel list to the population list that contains the fitness value for each solution.
    :param tournament_size: (int, optional): The number of individuals in each tournament. Defaults to 3.

    :return: selected_parents: (list): A list of solutions selected to be parents, based on their fitness.
    """
    selected_parents = []  # Empty list to store the selected parents.

    for _ in range(len(population)):
        # Randomly select 'tournament_size' individuals from the population for the tournament.
        tournament = random.sample(list(enumerate(population)), tournament_size)
        # Determine the individual with the lowest fitness score among the tournament participants.
        # The enumerate function adds an index to each solution, which is used to find its corresponding fitness.
        fittest_individual = min(tournament, key=lambda x: t_fitness[x[0]])
        # Add the fittest individual from the tournament to the list of selected parents.
        selected_parents.append(fittest_individual[1])

    return selected_parents


def two_point_crossover(parent1, parent2, problem_instance):
    """Executes a two-point crossover between two parent solutions to produce two offspring.
    This method selects two points within the solutions and swaps the segments between these points
    from one parent to the other, ensuring that item groups are not split.

    :param parent1: (list): The first parent solution.
    :param parent2: (list): The second parent solution.
    :param problem_instance: (dict): Contains details of the problem instance, including item weights and counts.

    :return: offspring1, offspring2: (tuple): A tuple containing two offspring solutions resulting from the crossover.

    Raises: ValueError: If the lengths of the parent solutions do not match.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parent length unequal.")

    # Calculate cumulative end points for item groups to ensure crossover points do not split any item group.
    item_group_end_points = [sum(count for _, count in problem_instance['items'][:i + 1])
                             for i in range(len(problem_instance['items']))]

    # Randomly select two unique points for crossover, ensuring they fall between item groups.
    crossover_point1, crossover_point2 = sorted(random.sample(item_group_end_points, 2))

    # Create the first offspring by combining the segment before the crossover point from parent1,
    # the segment between the two points from parent2, and the segment after the second point from parent1.
    offspring1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]

    # Create the second offspring by combining the segments in reverse order compared to offspring1.
    offspring2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]

    return offspring1, offspring2


def mutate(solution, mutation_rate, problem_instance):
    """Applies strategic mutation to a given solution by reassigning items to bins that have sufficient capacity,
    potentially lowering the overall number of bins used or improving the distribution of item weights across bins.

    :param solution: (list): The current solution to be mutated, where each element represents the bin assignment for an item.
    :param mutation_rate: (float): The probability of any given item being reassigned to a different bin.
    :param problem_instance: (dict): Contains the problem specifics, including bin capacity and item weights.

    :return: mutated_solution: (list): A new solution after mutation, with some items potentially reassigned to different bins.
    """
    # Create a copy of the solution to mutate so the original is unchanged.
    mutated_solution = solution.copy()
    # Calculate the current load in each bin based on the solution.
    bin_loads = calculate_bin_loads(mutated_solution, problem_instance)
    # Prepare a dictionary to track the maximum capacity for each bin.
    bin_capacities = {bin_number: problem_instance['bin_capacity'] for bin_number in bin_loads.keys()}

    # Calculate how much capacity is left in each bin after accounting for current loads.
    remaining_capacities = {bin_number: bin_capacities[bin_number] - load for bin_number, load in bin_loads.items()}

    # Create a list of all item weights, repeated by their count, to iterate over them.
    item_weights = [weight for weight, count in problem_instance['items'] for _ in range(count)]

    # Iterate over each item in the solution, considering its weight.
    for i, item_weight in enumerate(item_weights):
        # Apply mutation based on the mutation rate.
        if random.random() < mutation_rate:
            # Identify bins that can accommodate the item without exceeding capacity.
            suitable_bins = [bin_number for bin_number, remaining in remaining_capacities.items() if
                             remaining >= item_weight]

            if suitable_bins:
                # Choose one of the suitable bins randomly for this item.
                chosen_bin = random.choice(suitable_bins)
                # Assign the item to the chosen bin in the mutated solution.
                mutated_solution[i] = chosen_bin

                # Update the remaining capacity of the chosen bin to reflect the added item.
                remaining_capacities[chosen_bin] -= item_weight

    return mutated_solution


def genetic_algorithm(problem_instance, population_size, generations, mutation_rate, elitism_size):
    """Executes the genetic algorithm to solve the Bin Packing Problem (BPP), optimising the packing of items into bins.

    :param problem_instance: (dict): Details of the BPP, including item sizes and bin capacity.
    :param population_size: (int): The number of solutions in each generation.
    :param generations: (int): The total number of generations to run the algorithm for.
    :param mutation_rate: (float): Probability of mutating a given item in the solution.
    :param elitism_size: (int): Number of top solutions to carry over to the next generation unchanged.

    :return: best_solution, best_fitness (tuple): The best solution found across all generations, and its fitness score.
    """
    # Tracking the best fitness value and the average number of bins used across generations.
    best_fitness_over_generations = []
    average_bins_over_generations = []

    # Generate an initial population of solutions.
    population = initialise_population(problem_instance, population_size)

    # Evaluate the fitness of the initial population.
    fits = [fitness(individual, problem_instance) for individual in population]

    # Initialise variables to track the best solution found so far.
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        # Implement elitism by preserving the best solutions directly to the next generation.
        sorted_population = sorted(zip(population, fits), key=lambda x: x[1])
        elites = [individual for individual, _ in sorted_population[:elitism_size]]

        # Selecting parents for crossover based on their fitness.
        parents = tournament_selection(population, fits)

        # Generating offspring through crossover and mutation operations.
        offspring = []
        for i in range(0, len(parents), 2):
            # Ensure we have pairs of parents for crossover.
            if i + 1 < len(parents):
                offspring1, offspring2 = two_point_crossover(parents[i], parents[i + 1], problem_instance)
                offspring.extend([mutate(offspring1, mutation_rate, problem_instance),
                                  mutate(offspring2, mutation_rate, problem_instance)])
            else:
                # If an odd number of parents, mutate the last one without crossover.
                offspring.append(mutate(parents[i], mutation_rate, problem_instance))

        # Form the new population from the offspring and the elite solutions.
        population = offspring[:population_size - elitism_size] + elites

        # Re-evaluate the fitness of the new population.
        fits = [fitness(individual, problem_instance) for individual in population]

        # Update tracking information with the best fitness found in this generation.
        best_fitness_over_generations.append(min(fits))

        # Calculate and track the average number of bins used in this generation.
        total_bins_used = sum(len(calculate_bin_loads(ind, problem_instance)) for ind in population)
        average_bins = total_bins_used / population_size
        average_bins_over_generations.append(average_bins)

        # Update the best solution if a new best is found.
        for individual, fit in zip(population, fits):
            if fit < best_fitness:
                best_fitness = fit
                best_solution = individual

    return best_solution, best_fitness


def visualise_solution(solution, problem_instance, instance_number, output_folder="output_bbp_plots"):
    """Visualises the distribution of items across bins for a given solution.
    This function creates a bar chart showing how items are distributed across bins,
    illustrating the total weight of items in each bin.

    :param solution: (list): The solution to visualise, indicating bin assignments for each item.
    :param problem_instance: (dict): Contains details of the problem instance, including item sizes.
    :param instance_number: (int): The identifier for the problem instance, used in the plot title and filename.
    :param output_folder: (str, optional): The directory where the plot will be saved. Defaults to "output_bbp_plots".
    """
    # Total weight of items in each bin based on the solution.
    bin_loads = calculate_bin_loads(solution, problem_instance)
    # Extract bin numbers and loads for plotting.
    bins = list(bin_loads.keys())
    loads = list(bin_loads.values())

    # Checking the output directory exists, creating it if necessary.
    os.makedirs(output_folder, exist_ok=True)

    # Create bar chart with bins on the x-axis and loads on the y-axis.
    plt.figure(figsize=(10, 6))
    plt.bar(bins, loads, color='blue')
    plt.xlabel('Bin Number')
    plt.ylabel('Total Weight in Bin')
    plt.title(f'Distribution of Items Across Bins - Instance {instance_number}')

    # Construct the file path for saving the plot image.
    output_path = os.path.join(output_folder, f"bin_distribution_instance_{instance_number}.png")

    plt.savefig(output_path)
    plt.close()


def main():
    filename = "Binpacking.txt"
    population_size = 100  # Size of the population. Large enough for a diverse set of solutions.
    generations = 50  # Number of generations, Opportunity for solutions to improve.
    mutation_rate = 0.5  # Specifies the rate at which mutations occur within the population.
    elitism_size = 2  # Number of top solutions carried over unchanged to next generation.

    # Reads the problem instances from the specified file.
    problem_instances = read_instances(filename)

    for i, instance in enumerate(problem_instances):
        print(f"\nSolving problem instance {i + 1}...")

        # Execute the genetic algorithm on the current problem instance.
        best_solution, best_fitness = genetic_algorithm(
            instance,
            population_size,
            generations,
            mutation_rate,
            elitism_size
        )

        # Prints the best fitness score and the solution found by the genetic algorithm.
        print(f"Problem instance {i + 1}: Best Fitness = {best_fitness}")
        print(f"Best Solution: {best_solution}\n")

        # Additional analysis on the best solution to provide insights into its efficacy.
        analyse_solution(best_solution, instance)

        # Visual output.
        visualise_solution(best_solution, instance, i + 1)


if __name__ == "__main__":
    main()
