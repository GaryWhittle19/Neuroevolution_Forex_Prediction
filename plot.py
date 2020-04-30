# Externals
import pandas as pd     # Multi-dimensional arrays (DataFrame)
import copy             # For making copies of data
import plotly as pl     # Plotting library
import plotly.graph_objs as go
import plotly.express as px
import cufflinks as cf  # Works with plotly


def plot_ohlc(df, x_title, y_title, title, html_title, auto_open_html):
    """Plot OHLC DataFrame.
    :rtype: none
    :param df: the dataframe to be plotted
    :param x_title: x axis title
    :param y_title: y axis title
    :param title: title of the plot
    :param html_title: html filename
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    fig = df.iplot(asFigure=True, kind='ohlc', up_color='blue',down_color='red', xTitle=x_title,
                   yTitle=y_title, title=title)
    fig.write_html(html_title, auto_open=auto_open_html)


def plot_window_optimization(df, x_, y_, x_title, y_title, title, html_title, auto_open_html):
    """Plot window optimization results DataFrame.
    :rtype: none
    :param df: the dataframe to be plotted
    :param x_: x axis data
    :param y_: y axis data
    :param x_title: x axis title
    :param y_title: y axis title
    :param title: title of the plot
    :param html_title: html filename
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    layout = go.Layout(
        yaxis=dict(
            range=[-0.05, 0.05]
        )
    )

    fig = df.iplot(asFigure=True, x=x_, y=y_, mode='lines',
                   xTitle=x_title, yTitle=y_title, title=title, layout = layout)
    fig.write_html(html_title, auto_open=auto_open_html)


def plot_predictions(predictions_df, target_df, target_df_iloc, n_predicted_days, html_title, auto_open_html):
    """Plot predictions DataFrame.
    :rtype: none
    :param predictions_df: the dataframe holding predictions and their generation
    :param target_df: the dataframe holding target data to learn
    :param target_df_iloc: iloc values to use for locating target data
    :param n_predicted_days: number of days that are being predicted - used for array creation
    :param html_title: the title of the html file
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    final_dataframe = copy.deepcopy(pd.DataFrame(target_df[0:target_df_iloc]))  # Copy target data for plotting
    final_dataframe.rename(columns={0: 'Price'}, inplace=True)                  # Rename column

    for p in range(0, len(predictions_df.index), 1):
        final_dataframe['prediction'] = ''                  # Create blank to hold prediction

        x = predictions_df['generation'].iloc[p]            # Set relevant generation
        prediction = predictions_df['prediction'].iloc[p]   # Set prediction for the generation

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction    # Fill column
        final_dataframe.rename(columns={'prediction':
                                        ('Gen ' + str(x) + ' Prediction')}, inplace=True)               # Rename column

    if n_predicted_days == 1:
        drawing_mode = 'markers+lines'
    else:
        drawing_mode = 'lines'

    fig = final_dataframe.iloc[-n_predicted_days*4:].iplot(
        asFigure=True, mode=drawing_mode,
        xTitle='Date', yTitle='Price', title=html_title)
    fig.write_html(html_title, auto_open=auto_open_html)


def plot_test_predictions(non_evolved_test_predictions_df, evolved_test_predictions_df, target_df,
                          target_df_iloc, n_predicted_days, html_title, auto_open_html):
    """Plot the test predictions DataFrames.
    :rtype: none
    :param non_evolved_test_predictions_df: the dataframe holding generation 0's predictions
    :param evolved_test_predictions_df: the dataframe holding the best generation's predictions
    :param target_df: the dataframe holding target data to learn
    :param target_df_iloc: iloc values to use for locating target data
    :param n_predicted_days: number of days that are being predicted - used for array creation
    :param html_title: the title of the html file
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    final_dataframe = copy.deepcopy(pd.DataFrame(target_df[0:target_df_iloc]))  # Copy target data for plotting
    final_dataframe.rename(columns={0: 'Price'}, inplace=True)                  # Rename column

    for p in range(0, len(non_evolved_test_predictions_df.index), 1):
        final_dataframe['prediction'] = ''      # Create blank to hold prediction

        x = non_evolved_test_predictions_df['generation'].iloc[p]            # Set relevant generation
        prediction = non_evolved_test_predictions_df['prediction'].iloc[p]   # Set prediction for the generation

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction    # Fill column
        final_dataframe.rename(columns={'prediction':
                                        ('Gen ' + str(x) + ' Prediction '
                                         + str(p + 1))}, inplace=True)                                  # Rename column

    for p in range(0, len(evolved_test_predictions_df.index), 1):
        final_dataframe['prediction'] = ''      # Create blank to hold prediction

        x = evolved_test_predictions_df['generation'].iloc[p]               # Set relevant generation
        prediction = evolved_test_predictions_df['prediction'].iloc[p]      # Set prediction for the generation

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction    # Fill column
        final_dataframe.rename(columns={'prediction':
                                        ('Gen ' + str(x) + ' Prediction '
                                         + str(p + 1))}, inplace=True)                                  # Rename column

    if n_predicted_days == 1:
        drawing_mode = 'markers+lines'
    else:
        drawing_mode = 'lines'

    fig = final_dataframe.iloc[-n_predicted_days*4:].iplot(
        asFigure=True, mode=drawing_mode,
        xTitle='Date', yTitle='Price', title=html_title)
    fig.write_html(html_title, auto_open=auto_open_html)
