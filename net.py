# Internals
from pyESN import ESN   # Echo State Networks
# Externals
import numpy as np      # Numbers processing
import copy             # For making copies of data


# Neural Network Creation Function
def create_neural_network(input_, output_, reservoir_, spectral_, sparsity_, noise_, input_scale, random_, silent_):
    """Create an Echo State Network.
    Specify hyperparameters to use when setting up the network.
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
                noise=noise_,                   # Add noise to better model a dynamic system
                input_scaling=[input_scale],    # Scale is important - not too big as to wipe out past information
                random_state=random_,           # Random number generator
                silent=silent_)                 # Silent = False means we can see which stage ESN is at via print()
    return model


# Neural Network Copying Function
def copy_neural_network(target_network):
    """Copy an Echo State Network.
    Provide the Echo State Network to create a copy of.
    :rtype: pyESN.ESN
    :param target_network: network to copy
    :return: pyESN Echo State Network model
    """
    model = copy.deepcopy(target_network)
    return model


# Neural Network Mutation Function
def mutate_weights(weights_array, mutation_rate):
    """Mutate weights of an Echo State Network.
    Specify the mutation rate to use during mutation process. Provide an array of weights to mutate.
    :rtype: numpy.ndarray
    :param weights_array: the array of weights to be put through the mutation process
    :param mutation_rate: rate at which weights will evolve - if random float between 0 and 1 is less than this, mutate
    :return: the weights after being mutated based on mutation_rate
    """
    target_weights = weights_array          # Different array entries could be different layers/networks
    n_nodes = len(target_weights)           # Get number of nodes in layer

    for node_index in range(n_nodes):       # Cycle through the nodes in the layer
        node = target_weights[node_index]   # Assign weights for this particular node

        mu, sigma = 0, 0.15                 # Mean and standard deviation for gaussian distribution
        mutation = np.random.normal(mu, sigma, 1)           # Create a mutation value using gaussian distribution

        x = np.random.uniform(0, 1, 1)
        if x < mutation_rate:
            target_weights[node_index] = node + mutation    # Perform weight mutation

    mutated_weights = target_weights        # Target weights have been mutated

    return mutated_weights                  # Return.


def get_mse(prices, predictions):
    """Get the mean squared error of a network by analysing actual price and the network's corresponding predictions.
    Provide a prices array and a predictions array.
    :rtype: float64
    :param prices: the prices that the predictions aim to mimic
    :param predictions: the predictions
    :return: the mean-squared error of the combined predictions
    """
    mse = np.square(np.subtract(prices, predictions)).mean()
    return mse


# Return fitness of neural network.
    # Basis for fitness is difference between prediction and true value?
def get_fitness(prediction, true_value, prediction_range):
    # Calculate difference
    dif = abs(prediction - true_value)
