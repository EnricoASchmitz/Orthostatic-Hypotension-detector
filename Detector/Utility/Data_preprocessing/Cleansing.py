# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions for cleaning the raw data

# Imports
import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

from Detector.Utility.Plotting import plotting
from Detector.Utility.PydanticObject import DataObject

# Variables
Threshold = 2  # Threshold for minimal difference
logger = logging.getLogger(__name__)

def butter_low_pass_filter(data: Union[pd.Series, np.ndarray], cutoff: float, fs: int, order: int) \
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


def remove_unrealistic_values(df, data_object: DataObject, mini=20, maxi=250):
    # get only bp values
    bp_col = data_object.target_col[0]
    bp = df[bp_col].copy()
    bp_copy = bp.copy()
    # set all values which are very low or high to nan
    bp_copy[bp_copy < mini - 5] = np.nan
    bp_copy[bp_copy > maxi + 5] = np.nan
    # get the standard deviation from the filtered data
    ro = bp_copy.rolling(data_object.hz * 5, min_periods=1, center=True)
    std = ro.std().ffill().bfill()
    # calculate the upper and lower bound
    upper = np.nanquantile(bp_copy, 0.95) + std * 2
    lower = np.nanquantile(bp_copy, 0.05) - std * 2
    # replace values above or below the respective threshold
    plot = False
    if np.any(lower > bp) and np.any(bp < mini):
        bp[lower > bp] = np.nan
        print("lowerbound reached")
        plot = True
    if np.any(upper < bp) and np.any(bp > maxi):
        bp[upper < bp] = np.nan
        print("upperbound reached")
        plot = True
    if plot:
        plotting.simple_plot(bp, df[bp_col])
    df[bp_col] = bp
    return df
