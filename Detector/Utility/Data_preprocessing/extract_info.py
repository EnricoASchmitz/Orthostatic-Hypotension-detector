# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Function to extract input and output values

# Imports
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler

from Detector.Utility.Data_preprocessing.Cleansing import butter_low_pass_filter
from Detector.Utility.Data_preprocessing.Transformation import time_as_datetime
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters

Threshold = 1  # Threshold for minimal difference
logger = logging.getLogger(__name__)


def get_baseline(bp: pd.Series, start: float) -> float:
    """ Extract baseline from bp
    
    Args:
        bp: bp data (systolic or diastolic)
        start: index when subject is standing
        
    Returns:
        baseline value
    
    """
    baseline_length = Parameters.baseline_length.value
    # Get baseline bp of a window ending 10 seconds before standing up
    bl = bp[start - baseline_length:start - 10].mean()
    return bl


def get_drop(bp: pd.Series, baseline_bp: float, start: float) -> Tuple[float, float, float]:
    """ Extract drop within 30 seconds after standing up
    
    Args:
        bp: bp data to extract baseline
        baseline_bp: baseline bp
        start: index when subject is standing
    
    Returns:
        difference between minimum and baseline (drop), drop rate, drop index
    
    """
    # Get the minimum within 30 seconds after standing up
    minimum = bp[start:start + 30]

    # Get the index of the drop
    drop_index = np.round(minimum.idxmin() - start, 4)
    # if drop index = 0 we check 10 seconds before the standing up marker
    if drop_index <= 0:
        start = start - 10
        minimum = bp[start:start + 30]
        drop_index = np.round(minimum.idxmin() - start, 4)

    minimum_bp = minimum.min()

    # Get the drop amount
    drop = baseline_bp - minimum_bp
    # Get the drop rate
    drop_rate = minimum_bp / baseline_bp

    return drop, drop_rate, drop_index


def get_recovery(bp: pd.Series, baseline: float, start: float, window_start: int, window_end: int) -> float:
    """ Get recovery value within a window
    
    Args:
        bp: bp data
        baseline: baseline bp value
        start: index when subject is standing
        window_start: start of the window
        window_end: end of the window
    
    Returns:
        recovery within window
    """
    # get the mean of a certain time window
    mean = bp[start + window_start:start + window_end].mean()
    # extract mean from the baseline to get the recovery
    recovery = baseline - mean
    if mean is np.nan:
        recovery = get_recovery(bp, baseline, start, window_start - 10, window_end - 10)
    return recovery


def get_recovery_rate(bp: pd.Series, baseline: float, start: float) -> float:
    """ Get recovery rate at time point

    Args:
        bp: BP data
        baseline: BP baseline
        start: index when subject is standing

    Returns:
        Recovery rate at time

    """
    time = Parameters.time.value
    # Get recovery value at certain time after the drop, calculate by getting the mean of bp[time-10: time]
    bp_window = bp[start + time - 10:start + time]
    # get mean of the window
    bp_later = bp_window.mean()
    # get the recovery rate
    rr = bp_later / baseline
    return rr


def resample_hz(df: pd.Series, seconds: int = 1) -> Resampler:
    """ Resample data to seconds

    Args:
        df: Data to resample
        seconds: seconds to resample to

    Returns:
        Resample object

    """
    df = time_as_datetime(df)
    resample_object = df.resample(f"{seconds}s", label='right', closed='right')
    return resample_object


def get_x_values(x: dict, df: pd.Series, type_id: str) -> dict:
    """ Extract mean, std, max and min from df

    Args:
        x: dictionary to save data to
        df: data to extract values from
        type_id: name for the key

    Returns:
        Dictionary with added data

    """
    resample_object = resample_hz(df)
    x[f'{type_id}_mean'] = resample_object.mean()
    x[f'{type_id}_std'] = resample_object.std()
    x[f'{type_id}_max'] = resample_object.max()
    x[f'{type_id}_min'] = resample_object.min()
    return x


def get_y_values(par: dict, df: pd.Series, start: float, bp_type: str) -> dict:
    """ Extract parameters from df for output dataframe

    Args:
        par: dictionary with additional output data
        df: data to get the parameters from
        start: index when subject is standing
        bp_type: name of the current data

    Returns:
        Dictionary with the parameters

    """
    baseline = get_baseline(df, start)

    # Extract values
    drop, drop_rate, drop_index = get_drop(df, baseline, start)
    # save values to the dictionary
    par[f"{bp_type}_drop"] = drop
    par[f"{bp_type}_drop_index"] = drop_index
    par[f"{bp_type}_drop_rate"] = drop_rate

    # Calculate recover at certain time windows
    windows_reconstruction = [(10, 20), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (115, 125), (145, 155)]
    for recovery_window in windows_reconstruction:
        window_start, window_end = recovery_window
        point = int(np.array(recovery_window).mean())
        name = f"{bp_type}_recovery_{point}"
        par[name] = get_recovery(df, baseline, start, window_start, window_end)
    # Calculate drop per second
    par[f'{bp_type}_drop_per_sec'] = drop / drop_index
    # get recovery rate at given time
    par[f'{bp_type}_recovery_rate_{Parameters.time.value}'] = get_recovery_rate(df, baseline, start)
    return par


def get_full_curve(bp_dict: dict, bp_type: str, bp: pd.Series, start: float, seconds: int) -> dict:
    """ Get the full curve of the bp

    Args:
        bp_dict: dictionary with addition curves
        bp_type: Name of the column
        bp: BP data
        start: index when subject is standing
        seconds: seconds to resample to
    Returns:
        dictionary with the added curve

    """
    baseline_length = Parameters.baseline_length.value
    # Extract the full bp curve
    resampled_bp = resample_hz(bp, seconds).mean()
    resampled_bp.index = [time.timestamp() for time in resampled_bp.index]
    start_int = round(start, 0)
    # Get the bp up until 180 seconds into the future
    bp_dict[bp_type] = resampled_bp[start_int - baseline_length:start_int + 180]
    return bp_dict


def get_indices(sitting: pd.Series, stand: pd.Series, stop_index: float) -> Tuple[float, float, float]:
    """ Extract all indexes when subject transitions

    Args:
        sitting: True when sitting/supine, False while standing.
        stand: False when sitting/supine, True while standing.
        stop_index: index for our last repeat

    Returns:
        Index when challenge starts, index when subject is standing, index when challenge ends
    """
    # Get index when the repeat starts
    start_index = sitting[stop_index:].idxmax()
    # Get index when we stand up during the repeat
    stand_index = stand[start_index:].idxmax()
    # Get index when this repeats ends
    stop_index = stand[stand_index:].idxmin()
    # Make sure that we have a stop and start index
    if stand_index == stop_index:
        stop_index = stand.index[-1]
    return start_index, stand_index, stop_index


def extract_values(data_object: DataObject, df: pd.Series,
                   stand_index: float, stop_index: float,
                   seconds: int, future_steps: int,
                   warning: str, i: int = 0) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], dict]]:
    """ Extract values from df

    Args:
        data_object: data object containing information about the data setup
        df: data to extract values from
        stand_index: index when subject is standing
        stop_index: end of the challenge
        seconds: Seconds to resample to
        future_steps: number of future seconds to use for standing (used for output)
        warning: Message to give as warning
        i: recursive count, default = 0

    Returns:
        (input data BP, input data NIRS, full curve BP), dictionary with parameters
    """
    # Dicts
    par = {}
    x = {}
    x_nirs = {}
    bp_dict = {}

    starting_index = stand_index - Parameters.baseline_length.value

    # get values for systolic and diastolic
    for bp_type in data_object.target_col:
        # get the data
        bp_data = df[bp_type].copy().ffill().bfill()
        # get x values for corresponding data
        bp = bp_data[starting_index:stop_index].copy()
        smooth_bp = butter_low_pass_filter(bp, cutoff=0.2, fs=100, order=2)

        x = get_x_values(x, smooth_bp, bp_type)
        # get parameters about the standing part
        par = get_y_values(par, bp_data.copy(), stand_index, bp_type)
        drop_timestamp = par[f"{bp_type}_drop_index"]
        if drop_timestamp <= 1:
            # If the BP drop is within 1 second of standing up we assume wrong markers
            stand_index = stand_index - 1
            if drop_timestamp <= 0:
                print(warning)
                print(f"Drop before marker, {bp_type} index {drop_timestamp};")
                stand_index = stand_index - drop_timestamp
            if i < 10:
                return extract_values(data_object=data_object, df=df,
                   stand_index=stand_index, stop_index=stop_index,
                   seconds=seconds, future_steps=future_steps,
                   warning=warning, i = i+1)
            else:
                return None

        # Get the full bp during standing
        bp_dict = get_full_curve(bp_dict, bp_type, bp_data.copy(), stand_index, seconds)

    # get values for oxy and dxy
    for Nirs_type in data_object.nirs_col:
        nirs = df[Nirs_type][starting_index:stop_index].copy()
        nirs = nirs.ffill().bfill()
        smooth_nirs = butter_low_pass_filter(nirs, cutoff=0.5, fs=100, order=2)
        x_nirs = get_x_values(x_nirs, smooth_nirs, Nirs_type)

    return convert_dict(x, x_nirs, bp_dict, future_steps), par


def convert_dict(x: dict, x_nirs: dict,
                 bp_dict: dict, future_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Convert dictionaries to dataframe and then to numpy array

    Args:
        x: dictionary containing all BP data for input
        x_nirs: dictionary containing all NIRS data for input
        bp_dict: dictionary containing all full curve BP data for output
        future_steps: number of output time steps

    Returns:
        BP input data, NIRS input data, BP output data
    """
    baseline_length = Parameters.baseline_length.value
    standing_length = Parameters.standing_length.value
    x_df = pd.DataFrame(x).ffill().bfill()
    x_array = np.array(x_df)[:baseline_length + standing_length]

    x_nirs_df = pd.DataFrame(x_nirs).ffill().bfill()
    x_nirs_array = np.array(x_nirs_df)[:baseline_length + standing_length]

    y_curve = pd.DataFrame(bp_dict).ffill().bfill()
    y_curve_array = np.array(y_curve)[:future_steps + baseline_length]

    return x_array, x_nirs_array, y_curve_array


def make_datasets(data_object: DataObject, sub: str, info: dict, seconds: int,
                  lists: Tuple[list, list, list, list, list]) \
        -> Tuple[list, list, list, list, list]:
    """ Make dataframes for input and output

    Args:
        data_object: Object containing information about the data
        sub: ID of the subject
        info: Dictionary containing information about the subject
        seconds: Seconds to resample to
        lists: lists to add all the data to

    Returns:
        lists with added data
    """
    x_dataframes, x_oxy_dxy, y_curves, infs, parameters = lists
    df = info['Data'].copy()
    chal = info["Challenge"]
    h_stages = df['stage']
    # Get protocol
    sitting = h_stages.str.contains("start", case=False)
    stand = h_stages.str.contains("stand", case=False)
    stop_index = df.index[0]

    for repeat in range(0, 3):
        warning = f"Subject {sub}; challenge: {chal}; repeat: {repeat + 1}"

        start_index, stand_index, stop_index = get_indices(sitting, stand, stop_index)
        if start_index == stand_index:
            logger.error(warning)
            logger.error(f"Missing repeat: {repeat + 1}")
            continue
        # Save the patient info
        inf = {"ID": sub, "challenge": str(chal), "repeat": int(repeat)}
        inf.update(info["info"].copy())

        # calculate expected rows in the dataframe
        future_steps = int(Parameters.future_seconds.value / seconds)

        ext = extract_values(data_object=data_object, df=df,
                             stand_index=stand_index, stop_index=stop_index,
                             seconds=seconds, future_steps=future_steps,
                             warning=warning)
        if ext is not None:
            (x_array, x_nirs_array, y_curve_array), par = ext
        else:
            print(warning)
            print("Markers incorrect")
            continue

        if np.min(y_curve_array) < 0:
            print(warning)
            print("BP can't be lower than zero")
            continue
        elif y_curve_array[:, 0].min() < 20:
            print(warning)
            print("SBP can't be lower than 20")
            continue

        elif y_curve_array[:, 1].min() < 10:
            print(warning)
            print("DBP can't be lower than 10")
            continue

        # Make sure data is in the right format, or we skip the repeat
        baseline_length = Parameters.baseline_length.value
        standing_length = Parameters.standing_length.value
        if x_array.shape[0] == baseline_length + standing_length and \
                y_curve_array.shape[0] == future_steps + baseline_length:

            x_dataframes.append(x_array)
            nirs_mean = x_nirs_array.mean(axis=0)

            x_oxy_dxy.append(x_nirs_array - nirs_mean)
            y_curves.append(y_curve_array)
            infs.append(inf)
            parameters.append(par)
        elif x_array.shape[0] == baseline_length + standing_length and \
                y_curve_array.shape[0] != future_steps:
            logger.warning(warning)
            logger.warning(f"y curve not long enough; {y_curve_array.shape[0]}/{future_steps}")
        else:
            logger.warning(warning)
            logger.warning(f"X incorrect/ insufficient data")
    return x_dataframes, x_oxy_dxy, y_curves, infs, parameters
