# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions for cleaning the raw data

# Imports
import logging

import numpy as np

# Variables
import pandas as pd

from Detector.Utility.Data_preprocessing.Cleansing import butter_lowpass_filter
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
        start = start - 10
        minimum = BP[start:start + 30]
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
        recovery = get_recovery(BP, baseline, start, window_start - 10, window_end - 10, printing=False)
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


def get_indices(sitting, stand, stop_index):
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


def extract_values(data_object, df, stand_index, baseline_length, stop_index, time, warning, seconds, standing_length,
                   future_steps):
    # Dicts
    par = {}
    x = {}
    x_nirs = {}
    bp_dict = {}

    starting_index = stand_index - baseline_length

    # get values for systolic and diastolic
    for BP_type in data_object.target_col:
        # get the data
        BP_data = df[BP_type]
        # get x values for corresponding data
        bp = BP_data[starting_index:stop_index].copy()
        smooth_bp = butter_lowpass_filter(bp, cutoff=0.2, fs=100, order=2)
        x = get_x_values(x, smooth_bp, BP_type)
        # get parameters about the standing part
        par = get_y_values(par, BP_data.copy(), time, stand_index, BP_type)
        drop_timestamp = par[f"{BP_type}_drop_index"]
        if drop_timestamp <= 1:
            # If the BP drop is within 1 second of standing up we assume wrong markers
            stand_index = stand_index - 1
            if drop_timestamp <= 0:
                print(warning)
                print(f"Drop before marker, index {drop_timestamp};")
                stand_index = stand_index - drop_timestamp

            return extract_values(data_object, df, stand_index, baseline_length, stop_index, time, warning, seconds,
                                  standing_length, future_steps)

        # Get the full BP during standing
        bp_dict = get_full_curve(bp_dict, BP_type, BP_data.copy(), stand_index, stop_index, baseline_length, seconds)

    # get values for oxy and dxy
    for Nirs_type in data_object.nirs_col:
        nirs = df[Nirs_type][starting_index:stop_index].copy()
        smooth_nirs = butter_lowpass_filter(nirs, cutoff=0.5, fs=100, order=2)
        x_nirs = get_x_values(x_nirs, smooth_nirs, Nirs_type)

    return convert_dict(x, x_nirs, bp_dict, baseline_length, standing_length, future_steps), par


def convert_dict(x, x_nirs, bp_dict, baseline_length, standing_length, future_steps):
    # Convert dictionary to dataframe and then to numpy array
    x_df = pd.DataFrame(x).interpolate().fillna(0)
    x_array = np.array(x_df)[:baseline_length + standing_length]

    x_nirs_df = pd.DataFrame(x_nirs).interpolate().fillna(0)
    x_nirs_array = np.array(x_nirs_df)[:baseline_length + standing_length]

    y_curve = pd.DataFrame(bp_dict).interpolate()
    y_curve_array = np.array(y_curve)[:future_steps + baseline_length]
    return x_array, x_nirs_array, y_curve_array


def make_datasets(data_object, sub, info, baseline_length, standing_length, time,future,seconds, lists):
    x_dataframes, x_oxy_dxy, y_curves, infs, parameters = lists
    df = info['Data'].copy()
    chal = info["Challenge"]
    h_stages = df['stage']
    # Get protocol
    sitting = h_stages.str.contains("start", case=False)
    stand = h_stages.str.contains("stand", case=False)
    stop_index = df.index[0]

    for repeat in range(0, 3):
        warning = f"Warning: Subject {sub}; challenge: {chal}; repeat: {repeat + 1}"

        start_index, stand_index, stop_index = get_indices(sitting, stand, stop_index)
        if start_index == stand_index:
            print(warning)
            print(f"Missing repeat: {repeat + 1}")
            continue
        # Save the patient info
        inf = {"ID": sub, "challenge": str(chal), "repeat": int(repeat)}
        inf.update(info["info"].copy())

        # calculate expected rows in the daframe
        future_steps = int(future / seconds)

        (x_array, x_nirs_array, y_curve_array), par = extract_values(data_object,
                                                                     df,
                                                                     stand_index,
                                                                     baseline_length,
                                                                     stop_index,
                                                                     time,
                                                                     warning,
                                                                     seconds,
                                                                     standing_length,
                                                                     future_steps)

        # Make sure data is in the right format or we skip the repeat
        if x_array.shape[0] == baseline_length + standing_length and y_curve_array.shape[
            0] == future_steps + baseline_length:
            x_dataframes.append(x_array)
            x_oxy_dxy.append(x_nirs_array - x_nirs_array.mean(axis=0))
            y_curves.append(y_curve_array)
            infs.append(inf)
            parameters.append(par)
        elif x_array.shape[0] == baseline_length + standing_length and \
                y_curve_array.shape[0] != future_steps:
            print(warning)
            print(f"y curve not long enough; {y_curve_array.shape[0]}/{future_steps}")
        else:
            print(warning)
            print(f"X incorrect/ insufficient data")
    return x_dataframes, x_oxy_dxy, y_curves, infs, parameters