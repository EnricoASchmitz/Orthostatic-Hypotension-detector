# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions for cleaning the raw data

# Imports
import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

# Variables
from Detector.Utility.PydanticObject import DataObject

Threshold = 1  # Threshold for minimal difference
logger = logging.getLogger(__name__)


def identify_flatliners(target_array, future=10):
    # Calculate the slope to identify flat lines
    slopes = []
    # calculate slope between point BP_0 until BP_future
    # length of our total array
    length = len(target_array)
    for point in range(length):
        # get point 1
        point1 = (point, target_array[point])
        # calculate n points into the future, where n = future
        f_point = point + future
        # if we surpass the total length it is our end point
        if f_point >= length:
            f_point = length - 1
        point2 = (f_point, target_array[f_point])
        # add calculate slope for the point to the list
        slopes.append(slope(point1, point2))
    # to numpy
    y = np.array(slopes)
    # apply a threshold
    y[y < Threshold] = 0
    return y


def remove_flatliners(df: pd.DataFrame, data_object: DataObject, seconds_per_plateau: float = 1.5) -> pd.DataFrame:
    """ Remove flatliners from dataframe

    Args:
        df: dataframe
        data_object: Object containing all the needed information
        seconds_per_plateau: the number of seconds we expect a plateau to be

    Returns:
        dataframe without flatliners
    """
    # get only bp values
    target_array = np.array(df[data_object.target_col])

    # Calculate the slope between a point and the next, flatliners will have a slope close to 0
    y = identify_flatliners(target_array)

    # Find plateaus by reversing find_peaks
    df["signal"] = True
    peaks, peak_plateaus = find_peaks(- y, plateau_size=data_object.hz * seconds_per_plateau)
    if len(peak_plateaus['plateau_sizes']) == 0:
        logger.warning("No plateau found")

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
        while counter < int(data_object.hz * seconds_per_plateau):
            # get the next value
            try:
                value_r = np.array(target_array)[re_index + new_index]
            except IndexError:
                break
            # if value is lower than our minimum it is the new minimum
            if lowest_point_r["Value"] > value_r:
                lowest_point_r["index"] = re_index + new_index
                lowest_point_r["Value"] = value_r
            # get the next value
            value_l = np.array(target_array)[le_index - new_index]
            # if value is lower than our minimum it is the new minimum
            if lowest_point_l["Value"] > value_l:
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
    df = df.where(df["signal"])
    df.drop("signal", inplace=True, axis=1)

    return df


def slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        return 0
    return (y2 - y1) / (x2 - x1)


def butter_lowpass_filter(data, cutoff, fs, order):
    index = data.index
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return pd.Series(y, index=index)
