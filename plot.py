# Internals
from net import get_mse
import parameters
# Externals
import pandas as pd     # Multi-dimensional arrays (DataFrame)
import numpy as np
import copy             # For making copies of data
import plotly as pl     # Plotting library
import plotly.graph_objs as go
import cufflinks as cf  # Works alongside plotly


def plot_ohlc(df, x_title, y_title, title, html_title, auto_open_html):
    """Plot OHLC DataFrame.
    :rtype: none
    :param df: the DataFrame to be plotted
    :param x_title: x axis title
    :param y_title: y axis title
    :param title: title of the plot
    :param html_title: html filename
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    fig = df.iplot(asFigure=True, kind='ohlc',  # Create figure using iplot
                   up_color='blue', down_color='red',
                   xTitle=x_title, yTitle=y_title,
                   title=title)
    fig.write_html(html_title, auto_open=auto_open_html)  # Write the html to show final plot


def plot_window_optimization(df, x_, y_, x_title, y_title, title, html_title, auto_open_html):
    """Plot window optimization results DataFrame.
    :rtype: none
    :param df: the DataFrame to be plotted
    :param x_: x axis data
    :param y_: y axis data
    :param x_title: x axis title
    :param y_title: y axis title
    :param title: title of the plot
    :param html_title: html filename
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    layout = go.Layout(title=html_title,  # Create layout
                       xaxis_title='Training Window Size',
                       yaxis_title='MSE',
                       paper_bgcolor='rgb(245,246,249)',
                       plot_bgcolor='rgb(245,246,249)',
                       xaxis=dict(
                           linecolor='rgb(204, 204, 204)',
                           gridcolor='rgb(204, 204, 204)',
                           zerolinecolor='rgb(204, 204, 204)',
                       ),
                       yaxis=dict(
                           linecolor='rgb(204, 204, 204)',
                           gridcolor='rgb(204, 204, 204)',
                           zerolinecolor='rgb(204, 204, 204)',
                           range=[0.00, 0.05]
                       ),
                       )

    fig = df.iplot(  # Create figure using iplot
        asFigure=True, x=x_, y=y_, mode='lines',
        xTitle=x_title, yTitle=y_title,
        title=title, layout=layout)

    fig.write_html(html_title, auto_open=auto_open_html)  # Write the html to show final plot


def plot_predictions(predictions_df, target_df, target_df_iloc, prediction_extremes,
                     n_predicted_days, html_title, auto_open_html):
    """Plot predictions DataFrame.
    :rtype: none
    :param predictions_df: the DataFrame holding predictions and their generation
    :param target_df: the DataFrame holding target data to learn
    :param target_df_iloc: number of values required from target_data
    :param prediction_extremes: the most extreme y values from the predictions
    :param n_predicted_days: number of days that are being predicted - used for array creation
    :param html_title: the title of the html file
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    final_dataframe = copy.deepcopy(pd.DataFrame(target_df[0:target_df_iloc]))  # Copy target data for plotting
    final_dataframe.rename(columns={0: 'Price'}, inplace=True)  # Rename column

    if n_predicted_days == 1:  # Determine drawing mode
        drawing_mode = 'markers+lines'
    else:
        drawing_mode = 'lines'
    x_range = n_predicted_days * parameters.x_range_factor  # Determine x-range for plot
    prediction_extremes += [final_dataframe.tail(x_range).select_dtypes(include=[np.float64]).min()[0],
                            final_dataframe.tail(x_range).select_dtypes(include=[np.float64]).max()[0]]
    ymin, ymax = min(prediction_extremes), max(prediction_extremes)  # Determine y-range for plot

    for p in range(0, len(predictions_df.index), 1):
        final_dataframe['prediction'] = ''  # Create blank to hold prediction

        gen = predictions_df['generation'].iloc[p]  # Set relevant generation
        prediction = predictions_df['prediction'].iloc[p]  # Set prediction for the generation
        column_name = 'Gen ' + str(gen) + ' Prediction'  # Name the DataFrame column

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction  # Fill column
        final_dataframe.rename(columns={'prediction': column_name}, inplace=True)  # Rename column

    layout = go.Layout(title=html_title,  # Create layout
                       xaxis_title='Days',
                       yaxis_title='Price',
                       paper_bgcolor='rgb(245,246,249)',
                       plot_bgcolor='rgb(245,246,249)',
                       xaxis=dict(
                           linecolor='rgb(204, 204, 204)',
                           gridcolor='rgb(204, 204, 204)',
                           zerolinecolor='rgb(204, 204, 204)',
                           range=[target_df_iloc - x_range, target_df_iloc]
                       ),
                       yaxis=dict(
                           linecolor='rgb(204, 204, 204)',
                           gridcolor='rgb(204, 204, 204)',
                           zerolinecolor='rgb(204, 204, 204)',
                           range=[ymin, ymax]
                       ),
                       )

    fig = final_dataframe.iplot(  # Create figure using iplot
        asFigure=True, mode=drawing_mode,
        xTitle='Date', yTitle='Price',
        title=html_title, layout=layout)

    fig.write_html(html_title, auto_open=auto_open_html)  # Write the html to show final plot


def plot_test_predictions(non_evolved_test_predictions_df, evolved_test_predictions_df, target_df,
                          all_prediction_extremes, target_df_iloc, n_predicted_days, html_title, auto_open_html):
    """Plot the test predictions DataFrames.
    :rtype: none
    :param non_evolved_test_predictions_df: the DataFrame holding generation 0's predictions
    :param evolved_test_predictions_df: the DataFrame holding the best generation's predictions
    :param target_df: the DataFrame holding target data to learn
    :param all_prediction_extremes: the list of extremes values from the test predictions (max/min y values)
    :param target_df_iloc: number of values required from target_data
    :param n_predicted_days: number of days that are being predicted - used for array creation
    :param html_title: the title of the html file
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    final_dataframe = copy.deepcopy(pd.DataFrame(target_df[0:target_df_iloc]))  # Copy target data for plotting
    final_dataframe.rename(columns={0: 'Price'}, inplace=True)  # Rename column
    prices = final_dataframe['Price'].iloc[
             target_df_iloc - n_predicted_days:target_df_iloc]  # Create price array for getting MSE

    fig = go.Figure()  # Create the graph object
    if n_predicted_days == 1:  # Determine drawing mode
        drawing_mode = 'markers+lines'
    else:
        drawing_mode = 'lines'
    x_range = n_predicted_days * parameters.x_range_factor  # Determine x-range for plot
    all_prediction_extremes += [final_dataframe.tail(x_range).select_dtypes(include=[np.float64]).min()[0],
                                final_dataframe.tail(x_range).select_dtypes(include=[np.float64]).max()[0]]
    ymin, ymax = min(all_prediction_extremes), max(all_prediction_extremes)  # Determine y-range for plot

    # Plot predictions from non evolved network:
    for p in range(0, len(non_evolved_test_predictions_df.index), 1):
        final_dataframe['prediction'] = ''  # Create blank to hold prediction

        prediction = non_evolved_test_predictions_df['prediction'].iloc[p]  # Set prediction for the generation
        column_name = 'Initial Prediction ' + str(p + 1)  # Name the DataFrame column

        line_size = get_line_weight_by_mse(get_mse(prices, prediction))  # Get line width based on MSE

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction  # Fill column
        final_dataframe.rename(columns={'prediction': column_name}, inplace=True)  # Rename column
        fig.add_trace(go.Scatter(y=final_dataframe[column_name],  # Plot column
                                 mode=drawing_mode,
                                 name=column_name,
                                 line=dict(color='orangered', width=line_size)))

    # Plot predictions from evolved network:
    for p in range(0, len(evolved_test_predictions_df.index), 1):
        final_dataframe['prediction'] = ''  # Create blank to hold prediction

        gen = evolved_test_predictions_df['generation'].iloc[p]  # Set relevant generation
        prediction = evolved_test_predictions_df['prediction'].iloc[p]  # Set prediction for the generation
        column_name = 'Gen ' + str(gen) + ' Prediction ' + str(p + 1)  # Name the DataFrame column

        line_size = get_line_weight_by_mse(get_mse(prices, prediction))  # Get line width based on MSE

        final_dataframe.loc[final_dataframe.tail(n_predicted_days).index, 'prediction'] = prediction  # Fill column
        final_dataframe.rename(columns={'prediction': column_name}, inplace=True)  # Rename column
        fig.add_trace(go.Scatter(y=final_dataframe[column_name],  # Plot column
                                 mode=drawing_mode,
                                 name=column_name,
                                 line=dict(color='forestgreen', width=line_size)))

    fig.add_trace(go.Scatter(y=final_dataframe['Price'],  # Add price trace
                             mode=drawing_mode,
                             name='Price',
                             line=dict(color='darkorange', width=parameters.max_test_line_width)))

    fig.update_layout(title=html_title,  # Update layout
                      xaxis_title='Days',
                      yaxis_title='Price',
                      paper_bgcolor='rgb(245,246,249)',
                      plot_bgcolor='rgb(245,246,249)',
                      xaxis=dict(
                          linecolor='rgb(204, 204, 204)',
                          gridcolor='rgb(204, 204, 204)',
                          zerolinecolor='rgb(204, 204, 204)',
                          range=[target_df_iloc - x_range, target_df_iloc]
                      ),
                      yaxis=dict(
                          linecolor='rgb(204, 204, 204)',
                          gridcolor='rgb(204, 204, 204)',
                          zerolinecolor='rgb(204, 204, 204)',
                          range=[ymin, ymax]
                      ),
                      )

    fig.write_html(html_title, auto_open=auto_open_html)  # Write the html to show final plot


def get_line_weight_by_mse(mse):
    """
    :rtype: float64
    :param mse: mean squared error of prediction for relevant line
    :return: line size as calculated using width factor and MSE
    """
    line_width_reduction = abs(mse * parameters.test_line_width_factor)  # Calculate line width reduction using mse
    if line_width_reduction < parameters.max_test_line_width:  # If width reduction is less than max width
        line_size = parameters.max_test_line_width - line_width_reduction  # Perform the reduction
        return line_size  # Return the correct line width
    return 0


def plot_results_(standard_count, evolved_count, html_title, auto_open_html):
    """Plot the final results of the network training and testing process - show effectiveness as bar charts.
    :rtype: none
    :param standard_count: number of times standard ESN proved more effective than evolved networks
    :param evolved_count: number of times evolved ESN proved more effective than standard ESN
    :param html_title: title of the html file
    :param auto_open_html: boolean determines whether html will open automatically
    :return: none
    """
    colors = ['lightsalmon', 'lightgreen']
    result_vars = ['Standard ESN', 'Neuroevolved ESN']
    fig = go.Figure([go.Bar(x=result_vars, y=[standard_count, evolved_count], marker_color=colors)])
    fig.update_layout(title_text='Effectiveness Results Bar Chart')
    fig.write_html(html_title, auto_open=auto_open_html)
