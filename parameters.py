# Training and prediction
prediction_window = 7                                           # How many days in single prediction
# Window optimization
n_optimizer_predictions = 100                                   # Num of predictions to make during window optimization
smallest_possible_training_window = 775                         # Smallest training window test in optimization
largest_possible_training_window = 775                          # Largest training window test in optimization
training_window_test_increase = 1                               # The number to increase by between window tests
# Network
n_reservoir = 800                                               # Size of Echo State Network (ESN) hidden state
spectral_radius = 1.50                                          # Spectral radius to use for weights in ESNs
sparsity = 0.225                                                # Sparsity to use for weights in ESN
noise = 0.0125                                                  # Noise to use for training of ESN
input_scaling = 0.95                                            # Scale ESN input to hidden weights using this value
# Neuroevolution
n_generations = 20000                                           # Number of generations to run genetic algorithm for
mutation_rate = 0.50                                            # Mutation rate by which to mutate Echo State Networks
# HTML plotting
html_auto_show = False                                          # If true, display plotly HTMLs upon creation
