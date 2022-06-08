# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions for cleaning the raw data

# Imports
import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from Detector.Utility.Plotting import plotting
from Detector.Utility.PydanticObject import DataObject
# Variables
from Detector.enums import Parameters

logger = logging.getLogger(__name__)


def identify_flatliners(target_array: np.ndarray, future=20):
    target_array = np.round(target_array, 0)
    rolling_data = pd.Series(target_array).rolling(future, min_periods=1).apply(different_values)
    stat = get_scaled_statistics(rolling_data)
    y = np.ediff1d(target_array, to_begin=0)
    y[y < 0] = -1
    y[y > 0] = 1
    y = pd.Series(y).rolling(future, min_periods=1, center=True).mean()
    y = abs(y)
    y = np.mean([y, stat], axis=0)
    y = np.round(y, 0)
    return y


def different_values(x):
    unique_values = np.unique(x)
    return len(unique_values)


def get_scaled_statistics(rolling_data):
    data = rolling_data.bfill().ffill()
    data = np.abs(data)
    max = np.max(data)
    min = np.min(data)
    data = np.array([(x - min) / (max - min) for x in data])
    return data


def remove_flatliners(df: pd.DataFrame, data_object: DataObject, seconds_per_plateau: float = 1.2, plot=True) \
        -> pd.DataFrame:
    """ Remove flatliners from dataframe

    Args:
        df: dataframe
        data_object: Object containing all the needed information
        seconds_per_plateau: the number of seconds we expect a plateau to be
        plot: Plot before and after filter

    Returns:
        dataframe without flatliners
    """
    # get only bp values
    target_array = df[data_object.target_col].copy()
    target_array.dropna(axis=0, inplace=True)
    target_array = np.array(target_array).flatten()

    # Get flatliners 0 is flat 1 is not
    y = identify_flatliners(target_array)

    # Find plateaus by reversing find_peaks
    df["signal"] = True
    peaks, peak_plateaus = find_peaks(- y, plateau_size=[data_object.hz * seconds_per_plateau, data_object.hz * 10])
    ps = peak_plateaus['plateau_sizes']
    if len(ps) == 0:
        logger.warning("No plateau found")
    elif len(ps) > 20:
        indexes = np.argsort(ps)[-20:]
        peaks = peaks[indexes]
        for key, values in peak_plateaus.items():
            peak_plateaus[key] = values[indexes]
    # since sometimes a plateau will not end at the lowest value, we cut off until we are at the lowest value
    for i in range(len(peak_plateaus['plateau_sizes'])):
        le_index = peak_plateaus['left_edges'][i]
        re_index = peak_plateaus['right_edges'][i]
        # cut off until we reached the lowest point
        lowest_point_r = {"index": None, "Value": np.inf}
        lowest_point_l = {"index": None, "Value": np.inf}
        # rows in the plateau
        counter = 0
        new_index = 0
        # check one second to see if there is a value lower than our minimum
        while counter < int(data_object.hz) and re_index + new_index < len(target_array):
            # get the next value
            value_r = np.array(target_array)[re_index + new_index]
            # if value is lower than our minimum it is the new minimum
            if value_r != np.nan and lowest_point_r["Value"] > value_r:
                lowest_point_r["index"] = re_index + new_index
                lowest_point_r["Value"] = value_r
            # get the next value
            value_l = np.array(target_array)[le_index - new_index]
            # if value is lower than our minimum it is the new minimum
            if value_l != np.nan and lowest_point_l["Value"] > value_l:
                lowest_point_l["index"] = le_index - new_index
                lowest_point_l["Value"] = value_l
            # if it is not a minimum, continue
            else:
                counter += 1
            new_index += 1
        re_index = lowest_point_r["index"]
        le_index = lowest_point_l["index"]

        logger.debug(f"Found plateau between {round(df.index[le_index], 3)} and {round(df.index[re_index], 3)}")
        # set signal to false if we identify it as a plateau
        flatliners = df.index[le_index:re_index]
        df.at[flatliners, 'signal'] = False
    logger.warning(f"Removing {len(peak_plateaus['plateau_sizes'])} plateaus")

    # remove flatliners
    df[data_object.target_col] = df[data_object.target_col].where(df["signal"])
    df.drop("signal", inplace=True, axis=1)

    if plot:
        plotting.simple_plot(x=df.index, y=df[data_object.target_col[0]],
                             y2=[target_array, y], y2_name=["original", "y"])

    return df


def butter_low_pass_filter(data: Union[pd.Series, np.ndarray], cutoff: float, fs: float, order: int) \
        -> Union[pd.Series, np.ndarray]:
    """ Perform butter worth smoothing

    Args:
        data: The data to perform smoothing on
        cutoff: Hz cutoff
        fs: frequency of data (Hz)
        order: polynomial order

    Returns:
        Smoothed data

    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    if isinstance(data, pd.Series):
        index = data.index
        y = pd.Series(y, index=index)
    return y


def hampel_filter(input_series, window_size, n_sigmas=3):
    k = 1.4826  # scale factor for Gaussian distribution
    new_series = input_series.copy()

    # helper lambda function
    MAD = lambda x: np.median(np.abs(x - np.median(x)))

    rolling_median = input_series.rolling(window=2 * window_size, center=True).median()
    rolling_mad = k * input_series.rolling(window=2 * window_size, center=True).apply(MAD)
    diff = np.abs(input_series - rolling_median)

    np_array = np.array(diff > rolling_mad.multiply(n_sigmas)).flatten()
    indices = list(np.argwhere(np_array).flatten())
    new_series.iloc[indices] = rolling_median.iloc[indices]

    return new_series


def remove_unrealistic_values(df, data_object: DataObject,
                              mini=Parameters.minimal_BP.value, maxi=Parameters.maximal_BP.value):
    # get only bp values
    bp_col = data_object.target_col[0]
    bp = df[bp_col].copy()
    bp_copy = bp.copy()
    # set all values which are very low or high to nan
    bp_copy[bp_copy < mini] = np.nan
    bp_copy[bp_copy > maxi] = np.nan
    df[bp_col] = bp_copy
    return df
