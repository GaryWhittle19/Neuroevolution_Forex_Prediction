# Internals
from pyESN import ESN   # Echo State Networks
# Externals
import numpy as np      # Numbers processing
import copy             # For making copies of data


# Neural Network Creation Function
def create_neural_network(input_, output_, reservoir_, spectral_, sparsity_, noise_, input_scale, random_, silent_):
    """Create an Echo State Network.
    :rtype: pyESN.ESN
    :param input_: number of input units to use in ESN
    :param output_: number of output units to use in ESN
    :param reservoir_: number of hidden units to use in the ESN
    :param spectral_: the spectral radius - scale the hidden state as such that the biggest eigenvalue equals this
    :param sparsity_: sparsity of the weight connections (proportion of weights set to 0)
    :param noise_: noise to add to each neuron, helps reduce generalization error
    :param input_scale: scale of the input - smaller will leave more trace of previous timestep input during fitting
    :param random_: use for random_state when initializing network
    :param silent_: if true, print fitting data
    :return: pyESN Echo State Network model
    """
    model = ESN(n_inputs=input_,                # Inputs: ones
                n_outputs=output_,              # Outputs: predicted daily open
                n_reservoir=reservoir_,         # Size of reservoir
                spectral_radius=spectral_,      # More: longer range interactions and slowed decay of information
                sparsity=sparsity_,             # Just keep this small
                noise=noise_,                   # Add noise to better model a dynamic system - reduces generalisation
                input_scaling=[input_scale],    # Scale is important - not too big as to wipe out past information
                random_state=random_,           # Random number generator
                silent=silent_)                 # Silent = False means we can see which stage ESN is at via print()
    return model


# Neural Network Copying Function
def copy_neural_network(target_network):
    """Copy an Echo State Network.
    :rtype: pyESN.ESN
    :param target_network: network to copy
    :return: pyESN Echo State Network model
    """
    model = copy.deepcopy(target_network)
    return model


# Neural Network Mutation Function
def mutate_weights(weights_array, mutation_rate):
    """Mutate weights of an Echo State Network.
    :rtype: numpy.ndarray
    :param weights_array: the array of weights to be put through the mutation process
    :param mutation_rate: rate at which weights will evolve - if random float between 0 and 1 is less than this, mutate
    :return: the weights after being mutated based on mutation_rate
    """
    target_weights = weights_array  # Different array entries could be different layers/networks
    n_nodes = len(target_weights)  # Get number of nodes in layer

    for node_index in range(n_nodes):  # Cycle through the nodes in the layer
        node = target_weights[node_index]  # Assign weights for this particular node

        x = np.random.uniform(0, 1, 1)
        if x < mutation_rate:
            mu, sigma = 1, 0.1  # Mean and standard deviation for gaussian distribution
            mutation = np.random.normal(mu, sigma, 1)  # Create a mutation value using gaussian distribution
            target_weights[node_index] = node * mutation  # Perform weight mutation

    mutated_weights = target_weights  # Target weights have been mutated

    return mutated_weights  # Return mutated weights as an np.ndarray


def get_mse(prices, predictions):
    """Get the mean squared error of a network by analysing actual price and the network's corresponding predictions.
    :rtype: float64
    :param prices: the prices that the predictions aim to mimic
    :param predictions: the predictions
    :return: the mean-squared error of the combined predictions
    """
    mse = np.square(np.subtract(prices, predictions)).mean()  # Calculate MSE based on prediction/price
    return mse  # Return mean squared error as a float64


def perform_crossover(best_member, old_population, crossover_rate):
    """Perform crossover on the population after evolution.
    :rtype: list
    :param best_member: fittest member of old population
    :param old_population: the old population to be used for crossover
    :param crossover_rate: rate at which the fittest member will crossover
    :return: the new Echo State Network population after crossover
    """
    population_size = len(old_population)
    new_population = [0] * population_size  # Create new population array
    crossover_cutoff = int(crossover_rate * population_size)  # Calculate the cutoff for crossover of population

    for m in range(0, crossover_cutoff, 1):  # For first crossover_cutoff in population
        new_population[m] = copy.deepcopy(best_member)  # Crossover fittest member

    for m in range(crossover_cutoff, population_size, 1):  # For the rest of the population
        new_population[m] = copy.deepcopy(old_population[m])  # Allow the old population through

    return new_population  # Return the new population of Echo State Networks as a list


# Return fitness of neural network.
def get_fitness(esn, prices, prediction_window, n_fitness_predictions):
    """Get the fitness of a network based on an average mse from multiple predictions.
    :rtype: float64
    :param esn: the Echo State Network to evaluate fitness of
    :param prices: the array of prices the ESN is to predict
    :param prediction_window: number of days to predict
    :param n_fitness_predictions: number of predictions to make for creating MSE array
    :return: the fitness of the network (average MSE of n_fitness_predictions predictions)
    """
    mses = [0.0] * n_fitness_predictions
    for p in range(0, n_fitness_predictions, 1):
        prediction = esn.predict(np.ones(prediction_window))  # Predict a window with evolved weights

        evolved_predictions = np.ones(prediction_window)
        evolved_predictions[0:prediction_window] = prediction[:, 0]  # Place prediction in array

        mses[p] = get_mse(prices, evolved_predictions)

    return np.average(mses)  # Return average MSE (fitness) of network as a float64
