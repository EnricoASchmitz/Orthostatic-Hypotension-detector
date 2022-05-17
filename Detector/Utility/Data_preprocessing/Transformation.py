# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions to transform the raw input data

# Imports
from typing import Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from Detector.Utility.PydanticObject import DataObject, InfoObject


def time_as_datetime(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """ Convert index seconds to datetime column

    Args:
        df: Dataframe

    Returns:
         dataframe with datetime index
    """
    # convert seconds to TimedeltaIndex
    datetime = pd.TimedeltaIndex(pd.to_timedelta(df.index.get_level_values(0), 'seconds'))
    # convert TimedeltaIndex to datetime
    df.index = pd.to_datetime(datetime.view(np.int64))
    return df


def resample(df: pd.DataFrame, data_object: DataObject, info_object: InfoObject) -> \
        Tuple[Union[pd.Series, pd.DataFrame], DataObject]:
    """ Resample the dataframe

    Args:
        df: dataframe to be resampled
        data_object: information retrieved from the data
        info_object: information retrieved from the config

    Returns:
        resampled dataframe
    """
    df = time_as_datetime(df)
    # resample
    df = df.resample(f"{info_object.time_row}ms", kind="timestamp", origin="epoch", label='right',
                     closed='right').interpolate()
    # Revert datetime back to seconds
    df.index = [time.timestamp() for time in df.index]
    # change hz
    data_object.hz = int(1000 / info_object.time_row)
    # keep track of that we reindex our data
    data_object.reindex = True
    return df, data_object


def get_stolic(bp: pd.Series, data_object: DataObject, info_object: InfoObject) -> Tuple[pd.Series, DataObject]:
    """ Get Systolic or Diastolic data

    Args:
        bp: BP data, needs to be reversed for diastolic (-BP)
        data_object: object containing information about the data
        info_object: object containing configuration info

    Returns:
        Resampled Systolic or Diastolic data, data_object
    """
    bp_90 = bp.rolling(250, min_periods=1, center=True).quantile(0.90)
    bp_recenter = bp - bp_90
    peaks_index, _ = find_peaks(bp_recenter, height=(0, bp_recenter.max()))
    peaks = bp.iloc[peaks_index]
    # get same length as bp
    peaks[bp.index[0]] = peaks.iloc[0]
    peaks[bp.index[-1]] = peaks.iloc[-1]
    peaks = peaks.sort_index()

    stolic_bp, data_object = resample(peaks, data_object, info_object)
    return stolic_bp, data_object


def add_diastolic_systolic_bp(target_df: pd.DataFrame, data_object: DataObject, info_object: InfoObject) -> \
        Tuple[pd.DataFrame, type(InfoObject)]:
    """ Add systolic and diastolic bp to the bp dataframe

    Args:
        target_df: The dataframe containing the blood pressure
        data_object: Object containing all the needed information
        info_object: information from config

    Returns:
        dataframe with systolic and diastolic blood pressure, info_object
    """
    # Keep track of the names for diastolic and systolic column
    bp_name = data_object.target_col[0]
    dia_name = f"{bp_name}_diastolic"
    sys_name = f"{bp_name}_systolic"

    bp = target_df[bp_name]
    # get systolic values
    target_df[sys_name], data_object = get_stolic(bp, data_object, info_object)
    # get diastolic values
    target_df[dia_name], data_object = get_stolic(-bp, data_object, info_object)
    target_df[dia_name] = abs(target_df[dia_name])
    # save new target
    data_object.target_col = [sys_name, dia_name]

    return target_df, data_object


def scale3d(data: np.ndarray, data_object: DataObject) -> Tuple[np.ndarray, dict]:
    """ Scale 3D dataframe

    Args:
        data: 3D data to scale
        data_object: Object containing information about the data

    Returns:
        Rescaled data, scalers
    """
    scalers = {}
    for i in range(data.shape[1]):
        scalers[i] = data_object.scaler
        data[:, i, :] = scalers[i].fit_transform(data[:, i, :])
    return data, scalers


def reverse_scale3d(data: np.ndarray, scalers: dict[Any]) -> np.ndarray:
    """ Reverse 3d scaling

    Args:
        data: data to inverse scale
        scalers: scalers to use

    Returns:
        Inverse scaled data
    """
    for i in range(data.shape[1]):
        data[:, i, :] = scalers[i].inverse_transform(data[:, i, :])
    return data


def scale2d(data: Union[np.ndarray, pd.DataFrame], data_object: DataObject) \
        -> Tuple[Union[np.ndarray, pd.DataFrame], Any]:
    """ Scale 2D dataframe

    Args:
        data: 2D data to scale
        data_object: Object containing information about the data

    Returns:
        Rescaled data, scaler
    """
    scaler = data_object.scaler
    data = scaler.fit_transform(data)
    return data, scaler


def reverse_scale2d(data: Union[np.ndarray, pd.DataFrame], scaler: Any) -> Union[np.ndarray, pd.DataFrame]:
    """ Reverse 2d scaling

        Args:
            data: data to inverse scale
            scaler: scaler to use

        Returns:
            Inverse scaled data
        """
    return scaler.inverse_transform(data)
