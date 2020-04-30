# Internals
import numpy as np      # One-dimensional arrays (ndarray)
import pandas as pd     # Multi-dimensional arrays (DataFrame)


# Clean and return some data.
def return_univariate(filename, n_rows, target_column):
    """Return a numpy ndarray containing a single element from the specified .csv.
    Specify the .csv filename, number of rows and target column.
    :rtype: numpy.ndarray
    :param filename: the .csv file to read from
    :param n_rows: the last n_rows rows of data will be read
    :param target_column: which column to gather for univariate training/testing
    :return: univariate data (whichever column is passed as the target column)
    """
    # READ AND CLEAN DATA // -------------------- //
    data = pd.read_csv(filename)                                    # Read in the data
    # Create 1D Target Array to be used in Echo-State Networks
    target_data = data[target_column].iloc[-n_rows:]                # Create target dataframe
    data.drop_duplicates(keep=False)                                # Remove useless data (weekend closures...)
    target_values = target_data.values
    target_array = np.delete(target_values, [1], axis=1)
    return target_array


# Return a dataframe 'filename'.csv.
def return_multivariate(filename, n_rows, column_names, target_columns):
    """Return a pandas DataFrame containing multiple elements from the specified .csv.
    Specify the .csv filename, number of rows and target columns. Provide names of all columns for data cleaning.
    :rtype: numpy.ndarray
    :param filename: the .csv file to read from
    :param n_rows: the last n_rows rows of data will be read
    :param column_names: names of all columns in the .csv, disregarding the data column
    :param target_columns: which column to gather for univariate training/testing
    :return: multivariate data (whichever column are passed as the target columns)
    """
    # READ AND CLEAN DATA // -------------------- //
    data = pd.read_csv(filename)                                    # Read in the data
    data.date = pd.to_datetime(data.date, format='%Y.%m.%d')        # Format data for processing
    data = data.set_index(data.date)
    data = data[column_names]                                       # Rename columns after setting date as index
    data.drop_duplicates(keep=False)                                # Remove useless data (weekend closures...)
    data = data.iloc[-n_rows:]                                      # Locate the last n_rows entries
    target_data = data[target_columns]                              # Create target array
    return target_data
