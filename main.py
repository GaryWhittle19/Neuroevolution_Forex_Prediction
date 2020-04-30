# Internals
import data                             # Read and clean .csv data
import net                              # Create and modify Echo State Networks
import optimize                         # Echo-State Network hyperparameter optimization
import plot                             # Plot data
import parameters                       # User controlled parameters
# Externals
import copy                             # For making copies of data
import numpy as np                      # Numbers processing
import pandas as pd                     # Multi-dimensional arrays (DataFrame)

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
noise = parameters.sparsity                                     # Noise to use for training of ESN
input_scaling = parameters.input_scaling                        # Scale ESN input to hidden weights using this value

# Neuroevolution
n_neuroevolution_predictions = 1                                # Num of predictions to make during each generation
n_generations = parameters.n_generations                        # Number of generations to run genetic algorithm for
n_predicted_days = prediction_window \
                   * n_neuroevolution_predictions               # How many total predicted days in each generation


def run(iteration):
    """Runs an iteration: fit ESN, evolve a second ESN, plot the predictions of both networks to compare effectiveness.
    :rtype: none
    :param iteration: which iteration of program is running - iteration is used to seed the random state, ensure variety
    :return: none
    """
    # CREATE THE ECHO STATE NETWORK // ------------------------------------------------------------------------------ //
    rng = np.random.RandomState(iteration)                  # Seed the random state
    esn = net.create_neural_network(1, 1, n_reservoir,
                                    spectral_radius,
                                    sparsity,
                                    noise,
                                    input_scaling,
                                    rng, True)              # Create Echo-State Network

    # OPTIMIZE THE ECHO STATE NETWORK HYPERPARAMETERS // ------------------------------------------------------------ //
    global training_window_size     # Globalize training_window_size to set it to the optimal window
    if training_window_size == 0:       # If training window isn't set yet (is 0)
        training_window_size = get_best_training_window(prediction_window, n_optimizer_predictions,
                                                        esn, target_data,
                                                        smallest_possible_training_window,
                                                        largest_possible_training_window,
                                                        training_window_test_increase)
        global prices               # Globalize prices to set the prices that will be predicted each iteration
        prices = target_data[training_window_size:training_window_size + n_predicted_days]

    # USE EVOLUTIONARY ALGORITHM TO TRAIN ECHO STATE NETWORK // ----------------------------------------------------- //
    # Storage for prediction data
    non_evolved_prediction = np.ones(n_predicted_days)  # Store predictions before neuroevolution
    evolved_predictions = np.ones(n_predicted_days)     # Store predictions after neuroevolution
    # Predictions DataFrame to hold each evolution which improved prediction accuracy
    predictions_df = pd.DataFrame(columns=(['generation', 'prediction']))

    # // ---------------------------------------------- (GENERATION 0) ---------------------------------------------- //
    esn.fit(np.ones(training_window_size), target_data[0:training_window_size])     # Train network on data
    prediction = esn.predict(np.ones(prediction_window))                            # Predict a window
    non_evolved_prediction[0:prediction_window] = prediction[:, 0]                  # Add to array

    # For managing weights during evolution stage
    weights = copy.deepcopy(esn.W_out)
    # Make copies for storing initial/best weights to evaluate later
    initial_weights = copy.deepcopy(weights)
    best_weights = copy.deepcopy(weights)

    # Create a copy of generation 0's prediction as the last_good_prediction as it is currently the only prediction
    last_good_prediction = copy.deepcopy(non_evolved_prediction)                    # Stores most recent improvement
    best_generation = 0                                                             # Stores the best generation

    # Append the predictions DataFrame with generation 0's prediction
    iteration_df = predictions_df.append({'generation': 0, 'prediction': non_evolved_prediction}, ignore_index=True)
    predictions_df = copy.deepcopy(iteration_df)

    # // ------------------------------------- (GENERATIONS 1 TO N_GENERATIONS) ------------------------------------- //
    for generation in range(1, n_generations, 1):
        improvement = False

        # Perform mutation on feedback weights
        backup_weights = copy.deepcopy(weights)
        weights = net.mutate_weights(weights, parameters.mutation_rate)
        esn.W_out = copy.deepcopy(weights)

        # Calculate mean-squared error of non-evolved prediction
        before_mse = net.get_mse(prices, last_good_prediction)

        # Perform prediction using evolved network
        prediction = esn.predict(np.ones(prediction_window))
        evolved_predictions[0:0 + prediction_window] = prediction[:, 0]

        # Calculate mean-squared error of evolved network's prediction
        after_mse = net.get_mse(prices, evolved_predictions)

        # Calculate the difference in MSE
        mse_difference = after_mse - before_mse

        if mse_difference < 0:  # If MSE got smaller (good) - the MSE improved.
            last_good_prediction = copy.deepcopy(evolved_predictions)   # This is the most recent good prediction!
            iteration_df = predictions_df.append({'generation': (generation), 'prediction': last_good_prediction},
                                                 ignore_index=True)
            predictions_df = iteration_df

            best_generation = generation
            best_weights = copy.deepcopy(esn.W_out)

            # Print generation information
            print("Best generation: ", generation, "\nMSE before/after: ", before_mse, "/", after_mse,
                  "\nMSE difference: ", mse_difference, "\nImprovement: ", improvement)
        else:
            esn.W_out = copy.deepcopy(backup_weights)

    # Plot predictions from training
    plot.plot_predictions(predictions_df, target_data,
                          training_window_size + n_predicted_days, n_predicted_days,
                          'Training Predictions (' + str(iteration + 1) + ').html', parameters.html_auto_show)

    # TEST EVOLVED ECHO-STATE NETWORK ON NEW TARGET DATA // --------------------------------------------------------- //
    # Create DataFrame for prediction tests
    non_evolved_test_predictions_df = pd.DataFrame(columns=(['generation', 'prediction']))
    evolved_test_predictions_df = pd.DataFrame(columns=(['generation', 'prediction']))

    # NON-EVOLVED NETWORK PREDICTIONS
    esn.W_out = copy.deepcopy(initial_weights)
    # ---------------------------------------------------------------------------------------------------------------- #
    for p in range(1, 4, 1):
        # Perform prediction using non-evolved network
        new_prediction = esn.predict(np.ones(prediction_window))
        non_evolved_predictions = np.ones(n_predicted_days)
        non_evolved_predictions[0:0 + prediction_window] = new_prediction[:, 0]

        # prediction_number = '(' + str(p) + ')'
        temp_df = non_evolved_test_predictions_df.append({'generation': str(0),
                                                          'prediction': non_evolved_predictions},
                                                         ignore_index=True)
        non_evolved_test_predictions_df = copy.deepcopy(temp_df)

    # EVOLVED NETWORK PREDICTIONS
    if best_generation != 0:
        esn.W_out = copy.deepcopy(best_weights)
        # ------------------------------------------------------------------------------------------------------------ #
        for p in range(1, 4, 1):
            # Perform prediction using evolved network
            new_prediction = esn.predict(np.ones(prediction_window))
            evolved_predictions = np.ones(n_predicted_days)
            evolved_predictions[0:0 + prediction_window] = new_prediction[:, 0]

            temp_df = evolved_test_predictions_df.append({'generation': str(best_generation),
                                                          'prediction': evolved_predictions},
                                                         ignore_index=True)
            evolved_test_predictions_df = copy.deepcopy(temp_df)

    # PLOT TESTING PREDICTIONS
    current_iteration = iteration + 1
    plot.plot_test_predictions(non_evolved_test_predictions_df, evolved_test_predictions_df, target_data,
                               training_window_size + n_predicted_days, n_predicted_days,
                               'Testing Predictions (' + str(current_iteration) + ').html', parameters.html_auto_show)
    return


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
    plot.plot_ohlc(ohlc_to_plot, 'Date', 'Price', 'EUR-USD', 'OHLC.html', parameters.html_auto_show)


def get_best_training_window(prediction_window_to_optimize: int, n_predictions: int, esn: object,
                             target_data_for_optimization: np.ndarray,
                             min_possible_window: int, max_possible_window: int, training_window_increase: int) -> int:
    """Performs the training window optimization by testing each training window to find the best mean-squared error.
    :rtype: int
    :param prediction_window_to_optimize: the prediction window to optimize training window for
    :param n_predictions: number of windows to predict
    :param esn: the echo state network
    :param target_data_for_optimization: training and testing data
    :param max_possible_window: Max possible window can't be > (target data length - n_predicted_days)
    :param min_possible_window:  Min possible window to train on before predicting the prediction window
    :param training_window_increase: This is how much training window will increase between window tests
    :return: optimal training window
    """
    predicted_days = prediction_window_to_optimize * n_predictions              # Number of days in total to predict
    training_window_best, best_mse_ = optimize.optimize_training_window_(training_window_increase,
                                                                         min_possible_window, max_possible_window,
                                                                         prediction_window_to_optimize, predicted_days,
                                                                         esn, target_data_for_optimization)
    print("Optimal training window: ", training_window_best, "\n", "Mean squared error: ", best_mse_)
    return training_window_best


for i in range(1, 101, 1):
    print ("\nIteration: " + str(i) + " running.\n")
    run(i)
