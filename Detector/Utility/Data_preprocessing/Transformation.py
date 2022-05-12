# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions to transform the raw input data

# Imports
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

from Detector.Utility.PydanticObject import DataObject, InfoObject


def time_as_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert index seconds to datetime column

    Args:
        df: Dataframe

    Returns:
         dataframe with datetime column
    """
    # convert seconds to TimedeltaIndex
    datetime = pd.TimedeltaIndex(pd.to_timedelta(df.index.get_level_values(0), 'seconds'))
    # convert TimedeltaIndex to datetime
    df.index = pd.to_datetime(datetime.view(np.int64))
    return df


def resample(df: pd.DataFrame, data_object: DataObject, info_object: InfoObject) -> \
        Tuple[pd.DataFrame, DataObject]:
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


def get_stolic(BP: pd.Series, data_object, info_object) -> pd.Series:
    BP_Q3 = BP.rolling(250, min_periods=1, center=True).quantile(0.90)
    BP_recenterd = BP - BP_Q3
    peaks_index, _ = find_peaks(BP_recenterd, height=(0, BP_recenterd.max()))
    peaks = BP.iloc[peaks_index]
    # get same length as BP
    peaks[BP.index[0]] = peaks.iloc[0]
    peaks[BP.index[-1]] = peaks.iloc[-1]
    peaks = peaks.sort_index()
    stolic_BP, data_object = resample(peaks, data_object, info_object)
    return stolic_BP, data_object


def add_diastolic_systolic_bp(target_df: pd.DataFrame, data_object: DataObject, info_object: InfoObject) -> \
        Tuple[pd.DataFrame, type(InfoObject)]:
    """ Add systolic and diastolic BP to the BP dataframe

    Args:
        target_df: The dataframe containing the blood pressure
        data_object: Object containing all the needed information
        info_object: information from config

    Returns:
        dataframe with systolic and diastolic blood pressure, info_object
    """
    # Keep track of the names for diastolic and systolic column
    BP_name = data_object.target_col[0]
    dia_name = f"{BP_name}_diastolic"
    sys_name = f"{BP_name}_systolic"

    BP = target_df[BP_name]
    # get systolic values
    target_df[sys_name], data_object = get_stolic(BP, data_object, info_object)
    # get diastolic values
    target_df[dia_name], data_object = get_stolic(-BP, data_object, info_object)
    target_df[dia_name] = abs(target_df[dia_name])
    # save new target
    data_object.target_col = [sys_name, dia_name]

    return target_df, data_object


def scale_df(df: pd.DataFrame, columns: Optional[list] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """ Scale the dataframe columns

    Args:
        df: dataframe
        columns: Which columns to use, if None use all

    Returns:
        dataframe with scaled columns
    """
    # if no columns are specified we perform it on all columns
    if columns is None:
        columns = df.columns
    # get a minmax scaler
    scaler = MinMaxScaler()
    cdf = df[columns].copy()
    # scale data of the columns
    df[columns] = scaler.fit_transform(cdf)
    return df, scaler
