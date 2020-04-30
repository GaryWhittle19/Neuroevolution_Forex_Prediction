# Internals
import plot             # Plot data
# Externals
import numpy as np      # Numbers processing
import pandas as pd     # Multi-dimensional arrays (DataFrame)


# Optimize a target echo state network's hyperparameters
def optimize_training_window_(window_increase_, training_field_start, training_field_end,
                              prediction_window, n_predicted_days, esn, target_data_):
    """Calculate the optimal training window an Echo State Network can use when predicting a prediction window.
    Specify the window search parameters and prediction parameters. Provide Echo State Network and prediction target.
    :rtype: int, float64
    :param window_increase_: value by which the training window will increase with each iteration
    :param training_field_start: initial training window to test
    :param training_field_end: final training window to test
    :param prediction_window: window of prediction
    :param n_predicted_days: number of days to predict
    :param esn: the Echo State Network
    :param target_data_: the data to be trained and tested on
    :return: optimal training window size, corresponding MSE calculation
    """

    # Variables
    best_mse = 99999
    best_training = 0

    predictions = np.ones(n_predicted_days)     # Prediction array
    final_target = target_data_                 # The target data to use for training and testing
    training_df = pd.DataFrame(columns=(['window', 'mse']))

    for i in range (training_field_start, training_field_end + 1, window_increase_):
        # Create training and prediction data lengths
        train_length = i
        print("Testing ", train_length, " training window length.")

        # TEST WITH THE ECHO STATE NETWORK // -------------------- //
        for j in range(0, n_predicted_days, prediction_window):
            esn.fit(np.ones(train_length), final_target[j:train_length + j])    # Teacher-force output
            prediction = esn.predict(np.ones(prediction_window))                # Predict next (future) values
            predictions[j:j + prediction_window] = prediction[:, 0]             # Place prediction in predictions arr

        # Calculate mean-squared error
        act_tot = final_target[train_length:train_length + n_predicted_days]
        mse = np.square(np.subtract(act_tot, predictions)).mean()
        # Mean Squared Error
        if mse < best_mse:
            best_mse, best_training = mse, train_length
            print("Best MSE/training so far: ", best_mse, "/", best_training, "\n")

        iteration_df = training_df.append({'window' : i , 'mse' : mse} , ignore_index=True)
        training_df = iteration_df

    plot.plot_window_optimization(training_df, 'window', 'mse', 'Training window',
                                  'MSE', 'Window optimization', 'optimal_window.html')
    return best_training, best_mse
