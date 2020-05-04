# Internals
import data             # Read and clean .csv data
import net              # Create and modify Echo State Networks
import optimize         # Echo-State Network hyperparameter optimization
import plot             # Plot data
import parameters       # User controlled parameters
# Externals
import copy             # For making copies of data
import numpy as np      # Numbers processing
import pandas as pd     # Multi-dimensional arrays (DataFrame)

# How many tests to run
n_iterations = parameters.n_iterations
# Training and prediction windows
training_window_size = 0
prediction_window = parameters.prediction_window
# Window optimization
n_optimizer_predictions = parameters.n_optimizer_predictions
smallest_possible_training_window = parameters.smallest_possible_training_window
largest_possible_training_window = parameters.largest_possible_training_window
training_window_test_increase = parameters.training_window_test_increase
# Data
entries_required = (prediction_window * n_optimizer_predictions) \
                   + largest_possible_training_window           # Number of recent entries required from .csv
target_data = data.return_univariate('eu.csv',
                                     entries_required,
                                     ['open'])                  # Target data - data to train and test on
prices = 0                                                      # Stores prices to predict - set during execution
# Network
n_reservoir = parameters.n_reservoir                            # Size of Echo State Network (ESN) hidden state
spectral_radius = parameters.spectral_radius                    # Spectral radius to use for weights in ESNs
sparsity = parameters.sparsity                                  # Sparsity to use for weights in ESNs
noise = parameters.noise                                        # Noise to use for training of ESN
input_scaling = parameters.input_scaling                        # Scale ESN input to hidden weights using this value
# Neuroevolution
n_generations = parameters.n_generations                        # Number of generations to run genetic algorithm for
n_population = parameters.n_population                          # Number of networks in each generation
mutation_rate = parameters.mutation_rate                        # Rate at which to mutate the networks
crossover_rate = parameters.crossover_rate                      # Rate at which networks will crossover to next gen
n_fitness_predictions = parameters.n_fitness_predictions        # Number of predictions to test fitness upon
# Amount of tests per network after neuroevolution
n_test_predictions = parameters.n_test_predictions
# Initialize the variables for determining effectiveness
evolved_worse_count = 0
evolved_better_count = 0
total_non_evolved_average_mses = []
total_evolved_average_mses = []


def run(iteration):
    """Runs an iteration: fit ESN, evolve a second ESN, plot the predictions of both networks to compare effectiveness.
    :rtype: none
    :param iteration: which iteration of program is running - iteration is used to seed the random state, ensure variety
    :return: none
    """
    # CREATE THE ECHO STATE NETWORK // ------------------------------------------------------------------------------ //
    rng = np.random.RandomState(iteration)  # Seed the random state
    esn = net.create_neural_network(1, 1, n_reservoir,
                                    spectral_radius,
                                    sparsity,
                                    noise,
                                    input_scaling,
                                    rng, True)  # Create Echo-State Network

    # OPTIMIZE THE ECHO STATE NETWORK HYPERPARAMETERS // ------------------------------------------------------------ //
    global training_window_size  # Globalize training_window_size to set it to the optimal window
    if training_window_size == 0:  # If training window isn't set yet (is 0)
        training_window_size = get_best_training_window(prediction_window, n_optimizer_predictions,
                                                        esn, target_data,
                                                        smallest_possible_training_window,
                                                        largest_possible_training_window,
                                                        training_window_test_increase)
        global prices  # Globalize prices to set the prices that will be predicted each iteration
        prices = target_data[training_window_size:training_window_size + prediction_window]

        # Plot OHLC using plotly
        write_ohlc_html(entries_required, training_window_size + prediction_window)

    # // ------------------------------------------------ (BASE ESN) ------------------------------------------------ //
    # Predictions DataFrame to hold each evolution which improved prediction accuracy
    predictions_df = pd.DataFrame(columns=(['generation', 'prediction']))

    # Fit initial network
    esn.fit(np.ones(training_window_size), target_data[0:training_window_size])  # Train network on data

    # Set the benchmark network results
    best_generation = -1  # -1 is no evolution at all
    best_esn = net.copy_neural_network(esn)  # Set best esn
    best_mse = net.get_fitness(esn, prices,
                               prediction_window, n_fitness_predictions)  # Set best MSE

    # Perform prediction using initial network
    prediction = esn.predict(np.ones(prediction_window))  # Predict a window
    predictions = np.ones(prediction_window)  # For holding any predictions
    predictions[0:prediction_window] = prediction[:, 0]  # Place prediction in array
    best_prediction = copy.deepcopy(predictions)  # Set best prediction

    # Print initial network information
    print("Initial MSE before evolution: ", best_mse, "\n")

    # Append the predictions DataFrame with the initial best prediction
    iteration_df = predictions_df.append({'generation': -1, 'prediction': best_prediction}, ignore_index=True)
    predictions_df = copy.deepcopy(iteration_df)

    # USE EVOLUTIONARY ALGORITHM TO TRAIN ECHO STATE NETWORK // ----------------------------------------------------- //
    population_of_esns = [net.copy_neural_network(esn)] * n_population  # Create initial population

    # Create the extreme lists to hold extreme prediction values - used for y-range calculations during plotting
    training_prediction_extremes = [min(prediction)[0], max(prediction)[0]]

    # Perform neuroevolution
    for generation in range(0, n_generations, 1):
        print("Generation: ", generation)
        lowest_mse = best_mse
        for member_id, member in enumerate(population_of_esns):
            # Evolve the weights
            weights = copy.deepcopy(member.W_out)
            weights[0] = net.mutate_weights(weights[0], mutation_rate)
            member.W_out = copy.deepcopy(weights)

            # Get fitness of the member
            mse_fitness = net.get_fitness(member, prices, prediction_window, n_fitness_predictions)

            # If this member has the best fitness so far
            if mse_fitness < lowest_mse:
                # Set the benchmark network results
                best_esn = net.copy_neural_network(member)  # Stores the best member
                best_mse = mse_fitness  # And the best mse (fitness)
                # Perform prediction using this member
                prediction = member.predict(np.ones(prediction_window))
                prediction_array = np.ones(prediction_window)
                prediction_array[0:prediction_window] = prediction[:, 0]
                best_prediction = copy.deepcopy(prediction_array)
                # Store ID of this member
                best_member_id = member_id

        # Will be the case if any member improved the MSE from last generation
        if best_mse < lowest_mse:
            best_generation = generation
            # Print generation information
            print("\nBest generation: ", generation,
                  "\nBest member: ", best_member_id,
                  "\nMSE: ", best_mse,
                  "\nMSE before/after: ", lowest_mse, "/", best_mse, "\nMSE difference: ", best_mse - lowest_mse)
            # Append the predictions DataFrame with generation's best prediction
            iteration_df = predictions_df.append({'generation': generation,
                                                  'prediction': best_prediction},
                                                 ignore_index=True)
            predictions_df = copy.deepcopy(iteration_df)
            # Update extremes list for y-range of plot
            if min(prediction) < training_prediction_extremes[0]:
                training_prediction_extremes[0] = min(prediction)[0]
            if max(prediction) > training_prediction_extremes[1]:
                training_prediction_extremes[1] = max(prediction)[0]

        # Perform crossover
        population_of_esns = net.perform_crossover(best_esn, population_of_esns, crossover_rate)

    # Plot predictions from training
    plot.plot_predictions(predictions_df, target_data,
                          training_window_size + prediction_window, training_prediction_extremes, prediction_window,
                          'Training Predictions (' + str(iteration) + ').html', parameters.training_html_auto_show)

    # TEST EVOLVED ECHO-STATE NETWORK ON NEW TARGET DATA // --------------------------------------------------------- //
    # Create DataFrames for prediction tests
    dtypes = np.dtype([
        ('generation', str),
        ('prediction', np.float64)])
    columns_data = np.empty(0, dtype=dtypes)
    non_evolved_test_predictions_df = pd.DataFrame(columns_data)
    evolved_test_predictions_df = pd.DataFrame(columns_data)
    # Create lists to hold the test network MSE calculations
    non_evolved_mse_list = [0.0] * n_test_predictions
    evolved_mse_list = [0.0] * n_test_predictions
    # Create the extreme lists to hold extreme prediction values - used for y-range calculations during plotting
    non_evolved_prediction_extremes = [99.0, -99.0]
    evolved_prediction_extremes = [99.0, -99.0]

    # NON-EVOLVED NETWORK PREDICTIONS  // --------------------------------------------------------------------------- //
    for p in range(1, n_test_predictions, 1):
        # Perform prediction using non-evolved network
        prediction = esn.predict(np.ones(prediction_window))
        prediction_array = np.ones(prediction_window)
        prediction_array[0:0 + prediction_window] = prediction[:, 0]
        non_evolved_mse_list[p] = net.get_mse(prices, prediction_array)
        # Append the predictions DataFrame with non-evolved network prediction
        temp_df = non_evolved_test_predictions_df.append({'generation': str(-1),
                                                          'prediction': prediction_array},
                                                         ignore_index=True)
        non_evolved_test_predictions_df = copy.deepcopy(temp_df)
        # Update extremes list for y-range of plot
        if min(prediction) < non_evolved_prediction_extremes[0]:
            non_evolved_prediction_extremes[0] = min(prediction)[0]
        if max(prediction) > non_evolved_prediction_extremes[1]:
            non_evolved_prediction_extremes[1] = max(prediction)[0]

    # EVOLVED NETWORK PREDICTIONS // -------------------------------------------------------------------------------- //
    if best_generation != -1:
        for p in range(1, n_test_predictions, 1):
            # Perform prediction using evolved network
            neuroevolution_prediction = best_esn.predict(np.ones(prediction_window))
            prediction_array = np.ones(prediction_window)
            prediction_array[0:0 + prediction_window] = neuroevolution_prediction[:, 0]
            evolved_mse_list[p] = net.get_mse(prices, prediction_array)
            # Append the predictions DataFrame with evolved network prediction
            temp_df = evolved_test_predictions_df.append({'generation': str(best_generation),
                                                          'prediction': prediction_array},
                                                         ignore_index=True)
            evolved_test_predictions_df = copy.deepcopy(temp_df)
            # Update extremes list for y-range of plot
            if min(neuroevolution_prediction) < evolved_prediction_extremes[0]:
                evolved_prediction_extremes[0] = min(neuroevolution_prediction)[0]
            if max(neuroevolution_prediction) > evolved_prediction_extremes[1]:
                evolved_prediction_extremes[1] = max(neuroevolution_prediction)[0]

    # Update the prediction extremes array to pass into plotting function
    all_prediction_extremes = non_evolved_prediction_extremes
    if not evolved_prediction_extremes[0] == 99.0 and not evolved_prediction_extremes[1] == -99.0:
        all_prediction_extremes += evolved_prediction_extremes

    # PLOT TESTING PREDICTIONS // ----------------------------------------------------------------------------------- //
    plot.plot_test_predictions(non_evolved_test_predictions_df, evolved_test_predictions_df, target_data,
                               all_prediction_extremes, training_window_size + prediction_window, prediction_window,
                               'Testing Predictions (' + str(iteration) + ').html', parameters.testing_html_auto_show)

    # RETURN THE RESULTS OF THIS ITERATION // ----------------------------------------------------------------------- //
    return non_evolved_mse_list, evolved_mse_list


def write_ohlc_html(n_target_entries, n_training_data):
    """Creates a plotly graph of OHLC data.
    :rtype: none
    :param n_target_entries: how many entries to get from the OHLC .csv (example: 1000 is most recent 1000)
    :param n_training_data: how many entries to plot from the above returned target_entries (example: 100 is first 100)
    :return: none
    """
    # Plot OHLC using plotly
    raw_ohlc_data = data.return_multivariate('eu.csv', n_target_entries,
                                             ['open', 'high', 'low', 'close', 'volume'],
                                             ['open', 'high', 'low', 'close'])
    ohlc_to_plot = raw_ohlc_data.iloc[:n_training_data]
    plot.plot_ohlc(ohlc_to_plot, 'Date', 'Price', 'EUR-USD', 'OHLC.html', parameters.ohlc_html_auto_show)


def get_best_training_window(prediction_window_to_optimize: int, n_predictions: int, esn: object,
                             target_data_for_optimization: np.ndarray,
                             min_possible_window: int, max_possible_window: int, training_window_increase: int) -> int:
    """Performs the training window optimization by testing each training window to find the best mean-squared error.
    :rtype: int
    :param prediction_window_to_optimize: the prediction window to optimize training window for
    :param n_predictions: number of windows to predict
    :param esn: the echo state network
    :param target_data_for_optimization: training and testing data
    :param max_possible_window: Max possible window can't be > (target data length - prediction_window)
    :param min_possible_window:  Min possible window to train on before predicting the prediction window
    :param training_window_increase: This is how much training window will increase between window tests
    :return: optimal training window
    """
    predicted_days = prediction_window_to_optimize * n_predictions  # Number of days in total to predict
    training_window_best, best_mse_ = optimize.optimize_training_window_(training_window_increase,
                                                                         min_possible_window, max_possible_window,
                                                                         prediction_window_to_optimize, predicted_days,
                                                                         esn, target_data_for_optimization)
    print("Optimal training window: ", training_window_best, "\n", "Mean squared error: ", best_mse_, "\n")
    return training_window_best


def update_results(non_evolved_mse_list, evolved_mse_list):
    """Update the results which will be used in final bar chart plotting and printing of test results.
    :rtype: none
    :param non_evolved_mse_list:
    :param evolved_mse_list:
    :return: none
    """
    global evolved_worse_count
    global evolved_better_count
    if all(v == 0 for v in evolved_mse_list):  # If no evolution, less effective
        evolved_worse_count += 1
        total_non_evolved_average_mses.append(np.average(non_evolved_mses))
        print("\nITERATION " + str(i) + " SHOWED INEFFECTIVE NEUROEVOLUTION.\n")
    else:  # If there was an evolution
        if np.average(evolved_mse_list) >= np.average(non_evolved_mse_list):  # If evolved ESN performed worse
            evolved_worse_count += 1  # Increment ineffective count
            total_non_evolved_average_mses.append(np.average(non_evolved_mses))
            print("\nITERATION " + str(i) + " SHOWED INEFFECTIVE NEUROEVOLUTION.\n")
        else:  # If evolved ESN performed better
            evolved_better_count += 1  # Increment effective count
            total_evolved_average_mses.append(np.average(evolved_mses))
            print("\nITERATION " + str(i) + " SHOWED EFFECTIVE NEUROEVOLUTION.\n")


def plot_results():
    """Plot the final results bar chart comparing effectiveness of networks before/after neuroevolution
    :rtype: none
    :return: none
    """
    plot.plot_results_(evolved_worse_count, evolved_better_count,  # Plot results for analysis
                       'results.html', parameters.results_html_auto_show)


# Print execution information to console
print("Neuroevolution for Foreign Exchange Prediction Evaluation System.\n\n"
      " Target: ", prediction_window, " day(s).\n"
                                      " Hidden units: ", n_reservoir, "\n",
      "Spectral radius: ", spectral_radius, "\n",
      "Sparsity: ", sparsity, "\n",
      "Noise: ", noise, "\n",
      "Input scaling: ", input_scaling, "\n\n",

      "Generations: ", n_generations, "\n",
      "Population size: ", n_population, "\n",
      "Mutation rate: ", mutation_rate, "\n",
      "Crossover rate: ", crossover_rate, "\n\n",

      "Number of fitness calculation predictions: ", n_fitness_predictions, "\n",
      "Number of optimizer predictions: ", n_optimizer_predictions, "\n",
      "Number of test predictions per final network: ", n_test_predictions, "\n\n")

# Cycle through the iterations to train and test fitted/evolved Echo State Networks
for i in range(1, n_iterations + 1, 1):
    # Print current iteration
    print("\nIteration: " + str(i) + " running.\n")
    # Run program and return mse values for networks before/after neuroevolution
    non_evolved_mses, evolved_mses = run(i)
    # Update the test results
    update_results(non_evolved_mses, evolved_mses)

# Plot the test results
plot_results()
# Calculate and print the final test results
final_non_evolved_average_mse = np.average(total_non_evolved_average_mses)
final_evolved_average_mse = np.average(total_evolved_average_mses)
print(" Unsuccessful neuroevolution attempts: ", evolved_worse_count, "\n",
      "Successful neuroevolution attempts: ", evolved_better_count, "\n",
      "Average standard network MSE: ", final_non_evolved_average_mse, "\n",
      "Average neuroevolved network MSE: ", final_evolved_average_mse)
