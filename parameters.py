# Training and prediction
prediction_window = 7                                           # How many days in single prediction
# Window optimization
n_optimizer_predictions = 100                                   # Num of predictions to make during window optimization
smallest_possible_training_window = 25                          # Smallest training window test in optimization
largest_possible_training_window = 2000                         # Largest training window test in optimization
training_window_test_increase = 25                              # The number to increase by between window tests
# Network
n_reservoir = 400                                               # Size of Echo State Network (ESN) hidden state
spectral_radius = 1.50                                          # Spectral radius to use for weights in ESNs
sparsity = 0.2                                                  # Sparsity to use for weights in ESN
noise = 0.03                                                    # Noise to use for training of ESN
input_scaling = 1.0                                             # Scale ESN input to hidden weights using this value
# Neuroevolution
n_generations = 10                                              # Number of generations to run genetic algorithm for
n_population = 100
mutation_rate = 0.50                                            # Mutation rate by which to mutate Echo State Networks
crossover_rate = 0.25                                           # *100 = fittest % of population which will crossover
n_fitness_predictions = 100                                     # How many predictions when calculating fitness
# HTML plotting (NOTE: keep training/testing false as can severely impact performance)
training_html_auto_show = True                                  # If true display training plotly htmls upon creation
testing_html_auto_show = True                                   # If true display testing plotly htmls upon creation
ohlc_html_auto_show = True                                      # If true display OHLC html upon creation
window_html_auto_show = True                                    # If true display window optimization html upon creation
results_html_auto_show = True                                   # If true display results html upon creation
x_range_factor = 8                                              # Plot x-range will be n_predicted_days * this value
max_test_line_width = 1.5                                       # The maximum line width which
test_line_width_factor = 5000                                   # Test plot individual prediction line widths will be:
                                                                # max_line_width - abs(relevant_mse * line_width_factor)
                                                                # (Higher = more fadeout for bad MSE.
                                                                #  Lower = less fadeout for bad MSE.)
# Number of test predictions for each network
n_test_predictions = 10                                         # More will give better representation of effectiveness
# Number of iterations to run
n_iterations = 100                                              # How many tests to run
