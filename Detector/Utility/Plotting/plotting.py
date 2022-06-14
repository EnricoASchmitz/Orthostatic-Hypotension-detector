# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Plotting functions

# Imports
import logging
from typing import Union, Optional

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ceil
from plotly.graph_objs import Figure, Layout, Scatter, Histogram
from plotly.subplots import make_subplots

from Detector.Utility.Metrics.Losses import Loss
from Detector.enums import Parameters


def line_plot_with_stages(
        x: Union[np.array, str],
        y: Union[np.array, str],
        stand_markers: pd.DataFrame,
        title: str,
        df: Optional[pd.DataFrame] = None,
        **kwargs: any
) -> None:
    """ Line plot data using stand markers

    Args:
        x: x data to be plotted
        y: y data to be plotted
        stand_markers: dataframe containing marker timestamps
        title: string to use a title in the plot
        df: dataframe needed if x and y are column names

    Returns:
        None
    """
    fig = px.line(df, x=x, y=y, title=title, labels={"x": "time (seconds)"}, **kwargs)
    fig = plot_stages(fig, stand_markers)
    fig.show()


def scatter_plot_with_stages(
        x: Union[np.array, str],
        y: Union[np.array, str],
        stand_markers: pd.DataFrame,
        title: str,
        df: Optional[pd.DataFrame] = None,
        **kwargs: any
) -> None:
    """ Scatter plot data using stand markers

    Args:
        x: x data to be plotted
        y: y data to be plotted
        stand_markers: dataframe containing marker timestamps
        title: string to use a title in the plot
        df: dataframe needed if x and y are column names

    Returns:
         None
    """
    fig = px.scatter(df, x=x, y=y, title=title, labels={"x": "time (seconds)"}, **kwargs)
    fig = plot_stages(fig, stand_markers)
    fig.show()


def plot_stages(fig: Figure, stand_markers: pd.DataFrame) -> Figure:
    """ Add stages to a plot

    Args:
        fig: plotly figure to add the stages to needs to be indexed same way as stand_markers columns
        stand_markers: dataframe containing marker timestamps

    Returns:
        returns figure with stages
    """
    for index, row in stand_markers.iterrows():
        fig.add_vrect(x0=row["begin"], x1=row["end"],
                      annotation_text=index, annotation_position="top left",
                      fillcolor="green", opacity=0.25, line_width=0)
    return fig


def simple_plot(y: np.array, y2: Union[list, np.array] = None, x: np.ndarray = None, title: str = "plot",
                y2_name: Union[list, str] = "y2",
                mode: str = "lines") -> None:
    """ Wrapper for plotly line function

    Args:
        x: x data
        y: y data
        y2: y data for other lines (Optional)
        title: Title for the plot
        y2_name: Name for legend of y2
        mode: Line type

    Returns:
        None
    """
    fig = px.line(x=x, y=y, title=title)
    if y2 is not None:
        if isinstance(y2, list):
            for i, line in enumerate(y2):
                fig.add_scatter(x=x, y=line, cliponaxis=True, mode=mode, name=y2_name[i])
        else:
            fig.add_scatter(x=x, y=y2, cliponaxis=True, mode=mode, name=y2_name)
    fig.show()


def plot_prediction(target_name: str, target_index: int, prediction: np.ndarray, true: np.ndarray,
                    title: str, folder_name: str = None):
    """ Plot the prediction

    Args:
        folder_name: filename
        target_name: name to add to the plot
        target_index: column to plot
        true:  test data
        prediction: Predicted values
        title: title of the plot

    Returns:
        None
    """
    logger = logging.Logger(__name__)

    # get only the target column
    loss_f = Loss().get_loss_metric(Parameters.loss.value)
    loss_value = round(loss_f(true, prediction), 4)

    def get_correct_shape(data, target_i):
        if data.ndim > 2:
            data = data[:, -1, target_i]
        else:
            data = data[:, target_i]
        return data

    prediction = get_correct_shape(prediction, target_index)
    true = get_correct_shape(true, target_index)

    date_test = np.array(range(0, len(true)))

    if (np.amin(prediction) < 0) or (np.max(prediction) > 300):
        # Clip the prediction
        logger.warning("Values clipped")
        prediction = np.clip(prediction, a_min=0, a_max=300)

    trace2 = Scatter(
        x=date_test,
        y=prediction,
        mode="lines",
        name="Prediction",
    )
    trace3 = Scatter(
        x=date_test,
        y=true,
        opacity=0.5,
        mode="lines",
        name="Ground Truth"
    )

    plots = [trace2, trace3]

    layout = Layout(
        title=f"{title}, Loss {Parameters.loss.value}:{loss_value}",
        xaxis={"title": "Date"},
        yaxis={"title": f"{target_name}"}
    )
    fig = Figure(data=plots, layout=layout)
    if folder_name is not None:
        mlflow.log_figure(fig, f"figure/{folder_name}/prediction_{target_name}.html")
    else:
        fig.show()


def plot_comparison(model_name, info_df, names, rescaled_prediction, true, folder_name=None):
    c = 3
    r = int(ceil(true.shape[-1] / c))

    fig = make_subplots(rows=r, cols=c,
                        x_title="true",
                        y_title="prediction",
                        subplot_titles=names
                        )
    IDS = info_df.ID
    row_n = 1
    col_n = 1
    for col in range(rescaled_prediction.shape[-1]):
        fig.add_trace(
            Scatter(x=true[:, col],
                    y=rescaled_prediction[:, col],
                    name=names[col],
                    mode="markers",
                    showlegend=False,
                    hovertemplate="<b>%{text}</b><br>" +
                                  "True: %{x}<br>" +
                                  "Pred: %{y}<br>",
                    text=IDS),
            row=row_n, col=col_n
        )
        fig.add_shape(type="line",
                      x0=min(true[:, col]),
                      y0=min(true[:, col]),
                      x1=max(true[:, col]),
                      y1=max(true[:, col]),
                      line=dict(color="Red"),
                      row=row_n, col=col_n)

        col_n += 1
        if col_n == c + 1:
            col_n = 1
            row_n += 1
    fig.update_layout(height=500 * r, width=500 * c, title_text="Prediction vs True")
    if folder_name is not None:
        mlflow.log_figure(fig, f"{folder_name}/{model_name}_comparision.html")
    else:
        fig.show()


def plot_curves(sample, plot_index, reconstucted_curves_prediction, true_reconstucted_curve, target_index, BP_type,
                folder_name=None, streamlit_bool: bool = False):
    true = sample[:, target_index].copy()
    base_tuple = Parameters.baseline_tuple.value
    base_length = base_tuple[0] - base_tuple[1]
    baseline = true[:base_length].mean()
    true = true - baseline

    pred_curve = reconstucted_curves_prediction[plot_index]
    true_curve = true_reconstucted_curve[plot_index]
    x_list = list(np.arange(0, Parameters.future_seconds.value + 0.01, step=0.01))

    figure = Figure()
    figure.add_trace(Scatter(
        x=x_list,
        y=pred_curve[target_index],
        name=f'reconstructed predicted {BP_type}'
    ))
    figure.add_trace(Scatter(
        x=x_list,
        y=true_curve[target_index],
        name=f'reconstructed \'true\' {BP_type}'
    ))
    figure.add_trace(Scatter(
        x=list(range(-Parameters.baseline_tuple.value[0], Parameters.future_seconds.value, 1)),
        y=true,
        name=f'True {BP_type}'
    ))
    figure.update_layout(title_text=BP_type, xaxis_title="Seconds (After standing up)",
                         yaxis_title="Difference from baseline (mmHg)")

    if folder_name is not None:
        mlflow.log_figure(figure, f"{folder_name}/{BP_type}_parameter_prediction.html")
    elif streamlit_bool:
        return figure
    else:
        figure.show()


def plot_bars(col_names, true, pred, train, folder_name=None):
    for col, name in enumerate(col_names):
        fig = Figure(data=[Histogram(x=true[:, col], histnorm='probability', name="True Test")])
        fig.add_trace(
            Histogram(x=pred[:, col], histnorm='probability', name="Prediction Test")
        )
        fig.add_trace(
            Histogram(x=train[:, col], histnorm='probability', name="Train")
        )
        fig.update_layout(title_text=name)
        if folder_name is not None:
            mlflow.log_figure(fig, f"{folder_name}/{name}_parameter_bar_plot.html")
        else:
            fig.show()
