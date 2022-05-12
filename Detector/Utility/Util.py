# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Get information from the data

# Imports


# Variables
from typing import Union, Tuple, Any

import numpy as np
import pandas as pd

from Detector.Utility.PydanticObject import DataObject


def get_target_df(df: pd.DataFrame, data_object: DataObject) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Retrieve blood pressure from data frame. And make sure index is in datetime format.

    Args:
        df: Dataframe containing blood pressure
        data_object: Object containing all the needed information

    Returns:
        dataframe index with datetime, blood pressure dataframe index with datetime
    """
    # if we haven't reindex change index to datetime
    if not data_object.reindex:
        df = df.set_index("datetime")
    target = df[data_object.target_col].copy()
    # drop BP from the dataframe
    df.drop(data_object.target_col, axis=1, inplace=True)
    return df, target


def get_value(nested_value: Union[list, np.ndarray]) -> Union[int, str]:
    """ Get a nested value out of a list.

    Args:
        nested_value: nested list

    Returns:
        first value in the nested list
    """
    if isinstance(nested_value, (list, np.ndarray)):
        flattend_value = nested_value[0]
        nested_value = get_value(flattend_value)
    return nested_value


def get_markers(df: pd.DataFrame, markers_data: dict, prefix: str = 'stand') \
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
    df.loc[:, "stage"] = 'remove'
    markers_data = {k: v for k, v in sorted(markers_data.items(), key=lambda item: item[1])}
    # get the markers with when it happens
    for i, stage in enumerate(markers_data):
        stage_value = markers_data[stage]
        df.loc[stage_value:, 'stage'] = stage
        markers[stage] = stage_value
    # remove data points before the start and after the end
    df = df.loc[df.stage != "remove"].copy()
    df = df.loc[df.stage != "stop"].copy()
    # set stage a index
    df.set_index('stage', append=True, inplace=True)
    # create a dataframe of markers
    markers = pd.DataFrame.from_dict(markers, orient='index', columns=['begin'])
    markers.sort_values(by=['begin'], inplace=True)
    markers['end'] = markers['begin'].shift(periods=-1)
    # only get markers staring with stand
    result = [i for i in markers.index if prefix in i]
    stand_markers = markers.loc[result]
    stand_markers = stand_markers.clip(np.amin(df.index.get_level_values(0)), np.amax(df.index.get_level_values(0)))
    stand_markers = stand_markers[stand_markers['begin'] != stand_markers['end']]
    return df, stand_markers


def nan_helper(y: np.ndarray) -> Tuple[np.ndarray, Any]:
    """ Helper to handle indices and logical indices of NaNs.

    Args:
        y: array to check

    Returns:
         logical indices of NaNs, indexes
    """
    return np.isnan(y), lambda z: z.nonzero()[0]
