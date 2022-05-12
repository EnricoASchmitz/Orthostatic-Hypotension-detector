# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Outlier detection

# Imports
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from Detector.Utility.Data_preprocessing.Outlier_detection.OD_creator import OutlierDetector
from Detector.Utility.Data_preprocessing.Transformation import scale_df
from Detector.Utility.Plotting.plotting import scatter_plot_with_stages
from Detector.enums import ODAlgorithm


def outlier_detection(algo_name: Union[str, ODAlgorithm], df: pd.DataFrame, columns: List[str],
                      stand_markers: pd.DataFrame = None,
                      plot: bool = False):
    """ perform outlier detection with a specified algorithm

    Args:
        algo_name: outlier detection method
        df: dataframe containing the columns
        columns: columns to do outlier detection on
        stand_markers: dataframe containing marker timestamps
        plot: Do we want to plot the result

    Returns:
        dataframe with appended outlier column
    """
    # perform outlier detections
    df, anomaly_cols, algo_name = od(algo_name, df, columns)

    df["mean"] = df[anomaly_cols].mean(axis=1)
    # Check if dataframe already contains an outliers column
    if 'outliers' in df.columns:
        # adapt column to use in the next outlier detection calculation
        df['outliers'].replace({True: -1, False: 1}, inplace=True)
        df["mean"] = df[["mean", "outliers"]].mean(axis=1)

    # only set as outlier if outlier in most of the columns
    df["outliers"] = False
    outliers = df.index[np.where(df["mean"] < 0)]
    df.loc[outliers, "outliers"] = True
    df.drop(columns=anomaly_cols + ["mean"], inplace=True)

    if plot and (stand_markers is not None):
        scatter_plot_with_stages(
            df=df,
            x=df.index.get_level_values(0),
            y=columns,
            color="outliers",
            title=f"Outlier detection with {algo_name}",
            stand_markers=stand_markers
        )
    return df


def od(algo_name: Union[str, ODAlgorithm], df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str], str]:
    """ perform outlier detection with a specified algorithm

    Args:
        algo_name: outlier detection method.
        df: dataframe containing the columns.
        columns: columns to do outlier detection on.

    Returns:
        dataframe with appended outlier column, anomaly column names, used OD name
    """
    function = OutlierDetector.create_algorithm(algo_name, df=df)
    # perform outlier detection on provided columns
    for col in columns:
        data, scaler = scale_df(df, [col])
        df = function(data=data, col=col)

    anomaly_cols = [f"anomaly_{name}_{algo_name}" for name in columns]

    return df, anomaly_cols, algo_name
