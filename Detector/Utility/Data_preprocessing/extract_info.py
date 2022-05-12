# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions for cleaning the raw data

# Imports
import logging

import numpy as np

# Variables
from Detector.Utility.Data_preprocessing.Transformation import time_as_datetime

Threshold = 1  # Threshold for minimal difference
logger = logging.getLogger(__name__)


def get_baseline(BP, start):
    # Get baseline BP of a window of 30 seconds ending 10 seconds before standing up
    return BP[start - 40:start - 10].mean()


def get_drop(BP, baseline_BP, start):
    # Get the minimum within 30 seconds after standing up
    minimum = BP[start:start + 30]

    # Get the index of the drop
    drop_index = np.round(minimum.idxmin() - start, 4)
    # if drop index = 0 we check 10 seconds before the standing up marker
    if drop_index <= 0:
        minimum = BP[start - 10:start + 30]
        drop_index = np.round(minimum.idxmin() - start, 4)

    minimum_BP = minimum.min()

    # Get the drop amount
    drop = baseline_BP - minimum_BP
    # Get the droprate
    drop_rate = minimum_BP / baseline_BP

    return drop, drop_rate, drop_index


def get_recovery(BP, baseline, start, window_start, window_end, printing=False):
    # get the mean of a certain timewindow
    mean = BP[start + window_start:start + window_end].mean()
    # extract mean from the baseline to get the recovery
    recovery = baseline - mean
    if mean is np.nan:
        recovery = get_recovery(BP, baseline, start, window_start - 10, window_end - 10, printing=True)
    if printing:
        print(f"window = ({window_start},{window_end})")
    return recovery


def get_recovery_rate(BP, baseline, start, drop_index, time=60):
    # Get recovery value at certain time after the drop, calculate by getting the mean of BP[time-10: time]
    BP_later = BP[start + drop_index + time - 10:start + drop_index + time].mean()
    return BP_later / baseline


def resample_hz(df, seconds=1):
    df = time_as_datetime(df)
    resample_object = df.resample(f"{seconds}s", label='right', closed='right')
    return resample_object


def get_x_values(x, df, type_id):
    resample_object = resample_hz(df)
    x[f'{type_id}_mean'] = resample_object.mean()
    x[f'{type_id}_std'] = resample_object.std()
    x[f'{type_id}_max'] = resample_object.max()
    x[f'{type_id}_min'] = resample_object.min()
    return x


def get_y_values(par, df, time, start, BP_type):
    baseline = get_baseline(df, start)

    # Extract values
    drop, drop_rate, drop_index = get_drop(df, baseline, start)
    # save values to the dictionary
    par[f"{BP_type}_drop"] = drop
    par[f"{BP_type}_drop_index"] = drop_index
    par[f"{BP_type}_drop_rate"] = drop_rate

    # Calculate recover at certain timewindows
    windows_reconstuction = [(10, 20), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (115, 125), (145, 155)]
    for recovery_window in windows_reconstuction:
        window_start, window_end = recovery_window
        point = int(np.array(recovery_window).mean())
        name = f"{BP_type}_recovery_{point}"
        par[name] = get_recovery(df, baseline, start, window_start, window_end)
    # Calculate drop per second
    par[f'{BP_type}_drop_per_sec'] = drop / drop_index
    # get recovery rate at given time
    par[f'{BP_type}_recovery_rate_{time}'] = get_recovery_rate(df, baseline, start, drop_index, time)
    return par


def get_full_curve(BP_dict, BP_type, BP, start, end, baseline_length=40, seconds=1):
    # Extract the full BP curve
    resampled_BP = resample_hz(BP, seconds).mean()
    resampled_BP.index = [time.timestamp() for time in resampled_BP.index]
    start_int = round(start, 0)
    # Get the BP up until 180 seconds into the future
    BP_dict[BP_type] = resampled_BP[start_int - baseline_length:start_int + 180]
    return BP_dict
