# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script:
import os
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler
from plotly.graph_objs import Scatter, Figure
from pydantic import BaseModel
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import streamlit as st

class Parameters(Enum):
    """ Available parameters """
    iterations = 100
    n_trials = 5
    max_opt_hours = 24
    batch_size = 8
    validation_split = 0.2
    time_row_ms = 10
    default_units = 128
    time = 60
    baseline_tuple = (40, 10)
    rest_length = 40
    standing_length = 150
    future_seconds = 150
    recovery_times = [15, 20, 30, 40, 50, 60, 120, 150]
    loss = "mse"
    minimal_BP = 15
    maximal_BP = 300


class InfoObject(BaseModel):
    """ Object to save the needed information about the setup, taken from config """
    dataset: str
    file_loc: str
    smooth: bool
    model: str
    nirs_input: bool
    parameter_model: bool = None
    time_row: Optional[int] = Parameters.time_row_ms.value


def fetch_matlab_struct_data(matlab, imported_data=None, key="data",
                             level=1, parent_field=None, printing=False) -> Optional[dict]:
    """ Get the structure from a matlab file.

    Args:
        matlab: The mathlab file to read
        imported_data: dictionary to write info
        key: which key to extract data from
        level: needed for recursively extracting data
        parent_field: Used in recursively extracting data
        printing: Do we want to print data shapes (default = False)

    Returns:
        dictionary containing the data
    """
    if imported_data is None:
        imported_data = {}
    if isinstance(matlab, dict):
        if key not in matlab:
            data_key = [dict_key for dict_key in matlab if key in dict_key.lower()]
            try:
                for possible_key in data_key:
                    try:
                        return fetch_matlab_struct_data(matlab, key=possible_key)
                    except KeyError:
                        continue
            except IndexError:
                raise KeyError("Invalid key = {0:s}.".format(key))

        matlab_void = matlab[key]
    else:
        matlab_void = matlab

    if not isinstance(matlab_void, np.ndarray):
        return

    if matlab_void.shape == (1, 1):
        matlab_void = matlab_void[0, 0]
        if isinstance(matlab_void, np.void):
            mat_fields = list(matlab_void.dtype.fields.keys())
            if mat_fields:
                for field in mat_fields:
                    indent = "  " * level
                    child = matlab_void[field].squeeze()
                    if printing:
                        print(indent + "{0:s}: shape = {1}".format(field, child.shape))
                    if child.shape == (1, 1) and isinstance(child[0, 0], np.void):
                        fetch_matlab_struct_data(child, imported_data, level=level + 1, parent_field=field)
                    else:
                        if parent_field is not None:
                            key = parent_field + ":" + field
                        else:
                            key = field
                        imported_data[key] = child
    return imported_data


def _prepocessed_variables(python_dict: dict) -> Tuple[dict, dict, str, np.ndarray, int]:
    stages: list = [
        "start1_a",
        "endstand1_a",
        "stop1_a",
        "start2_a",
        "endstand2_a",
        "stop2_a",
        "start3_a",
        "endstand3_a",
        "stop3_a",
    ]
    markers = "marker"
    marker_times = python_dict[markers]
    marker_dict = {}
    for stage in stages:
        marker_dict[stage] = np.round(float(marker_times[stage]), 4)
    index_col = "nirs_time"
    # HZ
    hz_col = "fs_nirs"
    hz = int(python_dict[hz_col])

    # get index values
    start = 0
    step = 1 / hz
    num = len(python_dict["oxyvals"])
    index = np.arange(0, num) * step + start

    return python_dict, marker_dict, index_col, index, hz


def get_df(file, oxy_name, dxy_name):
    with st.spinner('Reading matfile...'):
        mat = loadmat(file)
        python_dict = fetch_matlab_struct_data(mat)
        data = {}
        cols = []
        python_dict, markers, index_col, index, hz = _prepocessed_variables(python_dict)
        for key, value in python_dict.items():
            if (key == oxy_name) or (key == dxy_name):
                # Average to obtain one signal
                left_long = slice(0, 3)
                right_long = slice(6, 9)
                left_long = np.mean(value[:, left_long], axis=1)
                right_long = np.mean(value[:, right_long], axis=1)
                if np.any(left_long >= 0) or np.isnan(left_long).all():
                    l_l = False
                else:
                    l_l = True
                if np.any(right_long >= 0) or np.isnan(right_long).all():
                    r_l = False
                else:
                    r_l = True
                if l_l and r_l:
                    value = np.mean(np.array([left_long, right_long]), axis=0)
                elif l_l and not np.isnan(left_long).all():
                    value = left_long
                elif r_l and not np.isnan(right_long).all():
                    value = right_long
                else:
                    value = np.mean(np.array([left_long, right_long]), axis=0)
                data[f"{key}_long"] = value
                cols.append(f"{key}_long")

        length_to_use = len(index)
        for key, value in data.items():
            if length_to_use > len(value):
                length_to_use = len(value)

        if length_to_use != len(index):
            index = index[:length_to_use]
            for key, value in data.items():
                data[key] = value[:length_to_use]

        df = pd.DataFrame(data, columns=cols, index=np.round(index, 3))
        if df.isnull().all(axis=0).any():
            raise ValueError("Data contains only NaN")
        elif df.empty:
            raise ValueError("File doesn't contain specified columns")
    return df, markers, hz


def get_clicks(clicking, remove=False):
    file_name = "counter.txt"
    try:
        with open(file_name, "r") as f:
            a = f.readline()  # starts as a string
            a = 0 if a == "" else int(a)  # check if its an empty string, otherwise should be able to cast using int()
    except FileNotFoundError:
        a = 0
    if clicking:
        a += 1
        with open(file_name, "w") as f:
            f.truncate()
            f.write(f"{a}")
    if remove:
        os.remove(file_name)
    return a


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


def get_markers(df: pd.DataFrame, markers_data: dict, prefix: str = "stand") \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Retrieve timestamps for markers

    Args:
        df: dataframe
        markers_data: dict containing timestamps with marker name as key
        prefix: which stage do we want (default= stand)

    Returns:
        dataframe with stage index, timestamps when marker happens
    """
    markers = {}
    df.loc[:, "stage"] = "remove"
    markers_data = {k: v for k, v in sorted(markers_data.items(), key=lambda item: item[1])}
    # get the markers with when it happens
    for i, stage in enumerate(markers_data):
        stage_value = markers_data[stage]
        df.loc[stage_value:, "stage"] = stage
        markers[stage] = stage_value
    # remove data points before the start and after the end
    df = df.loc[df.stage != "remove"].copy()
    df = df.loc[df.stage != "stop"].copy()
    # create a dataframe of markers
    markers = pd.DataFrame.from_dict(markers, orient="index", columns=["begin"])
    markers.sort_values(by=["begin"], inplace=True)
    markers["end"] = markers["begin"].shift(periods=-1)
    # only get markers staring with stand
    result = [i for i in markers.index if prefix in i]
    stand_markers = markers.loc[result]
    stand_markers = stand_markers.clip(np.amin(df.index.get_level_values(0)), np.amax(df.index.get_level_values(0)))
    stand_markers = stand_markers[stand_markers["begin"] != stand_markers["end"]]
    return stand_markers


def scale3d(data: np.ndarray, scalers) -> np.ndarray:
    """ Scale 3D dataframe

    Args:
        data: 3D data to scale
        data_object: Object containing information about the data

    Returns:
        Rescaled data, scalers
    """
    for i in range(data.shape[1]):
        data[:, i, :] = scalers[i].transform(data[:, i, :])
    return data


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
    x[f"{type_id}_mean"] = resample_object.mean()
    x[f"{type_id}_std"] = resample_object.std()
    x[f"{type_id}_max"] = resample_object.max()
    x[f"{type_id}_min"] = resample_object.min()
    return x


def resample_hz(df: pd.Series, seconds: int = 1) -> Resampler:
    """ Resample data to seconds

    Args:
        df: Data to resample
        seconds: seconds to resample to

    Returns:
        Resample object

    """
    df = time_as_datetime(df)
    resample_object = df.resample(f"{seconds}s", label="right", closed="right")
    return resample_object


def time_as_datetime(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """ Convert index seconds to datetime column

    Args:
        df: Dataframe

    Returns:
         dataframe with datetime index
    """
    # convert seconds to TimedeltaIndex
    datetime = pd.TimedeltaIndex(pd.to_timedelta(df.index.get_level_values(0), "seconds"))
    # convert TimedeltaIndex to datetime
    df.index = pd.to_datetime(datetime.view(np.int64))
    return df


def get_dataset(df, stand_markers, logger):
    x_oxy_dxy = []
    h_stages = df["stage"]
    # Get protocol
    sitting = h_stages.str.contains("start", case=False)
    stand = h_stages.str.contains("stand", case=False)
    df.drop("stage", axis=1, inplace=True)
    stop_index = df.index[0]
    x_nirs = {}
    for repeat in range(len(stand_markers)):
        start_index, stand_index, stop_index = get_indices(sitting, stand, stop_index)
        if start_index == stand_index:
            logger.error(f"Missing repeat: {repeat + 1}")
            continue
        # calculate expected rows in the dataframe
        x_nirs_array = extract_nirs(df, x_nirs, stand_index, stop_index)
        # Make sure data is in the right format, or we skip the repeat
        rest_length = Parameters.rest_length.value
        standing_length = Parameters.standing_length.value
        if x_nirs_array.shape[0] == rest_length + standing_length:
            nirs_mean = x_nirs_array.mean(axis=0)
            x_oxy_dxy.append(x_nirs_array - nirs_mean)
        else:
            raise ValueError("wrong")
    return x_oxy_dxy


def extract_nirs(df, x_nirs, stand_index, stop_index):
    starting_index = stand_index - Parameters.rest_length.value
    starting_index = round(starting_index, 2)
    # get values for oxy and dxy

    for Nirs_type in df.columns:
        nirs = df[Nirs_type][starting_index:stop_index].copy()
        nirs = nirs.ffill().bfill()
        # parameters: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2818571/
        smooth_nirs = butter_low_pass_filter(nirs, cutoff=0.01, fs=0.5, order=2)
        x_nirs = get_x_values(x_nirs, smooth_nirs, Nirs_type)

    rest_length = Parameters.rest_length.value
    standing_length = Parameters.standing_length.value

    x_nirs_df = pd.DataFrame(x_nirs).ffill().bfill()
    x_nirs_array = np.array(x_nirs_df)[:rest_length + standing_length]
    return x_nirs_array


def create_curve(drop, drop_time, BP_timepoints):
    BP = [0, drop] + list(BP_timepoints.values())
    times = [0, drop_time] + list(BP_timepoints.keys())

    data = pd.Series(BP, index=times)
    # convert seconds to TimedeltaIndex
    datetime = pd.TimedeltaIndex(pd.to_timedelta(data.index, 'seconds'))
    # convert TimedeltaIndex to datetime
    data.index = pd.to_datetime(datetime.view(np.int64))
    data = data.resample("10ms").mean().interpolate()
    data.index = [time.timestamp() for time in data.index]
    data.sort_index(inplace=True)
    return np.array(data)


def get_info(y, index, first_col, timesteps):
    column_number = 0
    baseline = 0

    drop_decrease = y[index, column_number + first_col]
    column_number += 1

    drop = -drop_decrease

    drop_time_predicted = y[index, column_number + first_col]
    column_number += 1

    drop_time = drop_time_predicted
    if drop_time < 0:
        drop_time = 0.01

    BP_timepoints = {}
    for step in timesteps:
        BP_timepoints[step] = baseline - y[index, column_number + first_col]
        column_number += 1

    return baseline, drop, drop_time, BP_timepoints


def parameters_to_curve(y, index, timesteps):
    curves = []
    columns = int(y.shape[-1] / 2)
    baseline, SBP_drop, SBP_drop_time, SBP_timepoints = get_info(y, index, 0, timesteps)
    curves.append(create_curve(SBP_drop, SBP_drop_time, SBP_timepoints))
    baseline, DBP_drop, DBP_drop_time, DBP_timepoints = get_info(y, index, columns, timesteps)
    curves.append(create_curve(DBP_drop, DBP_drop_time, DBP_timepoints))
    return np.array(curves)


def make_curves(prediction, reconstruct_params, steps):
    # resample the needed prediction parameters to 100Hz to get the curve
    reconstruct_prediction = np.array(prediction[reconstruct_params])
    reconstucted_curves_prediction = []
    for i in range(len(prediction)):
        pred_reconstruction = parameters_to_curve(reconstruct_prediction, i, steps)
        reconstucted_curves_prediction.append(pred_reconstruction)
    return np.array(reconstucted_curves_prediction)


def plot_curve(plot_index, reconstucted_curves_prediction, target_index, BP_type):
    pred_curve = reconstucted_curves_prediction[plot_index]
    x_list = list(np.arange(0, Parameters.future_seconds.value + 0.01, step=0.01))

    figure = Figure()
    figure.add_trace(Scatter(
        x=x_list,
        y=pred_curve[target_index],
        name=f'reconstructed predicted {BP_type}'
    ))
    figure.update_layout(title_text=BP_type, xaxis_title="Seconds (After standing up)",
                         yaxis_title="Difference from baseline (mmHg)")

    return figure
