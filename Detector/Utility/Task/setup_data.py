# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Function to get the data in the correct state

# Imports
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# for data types
from Detector.Utility.Models.abstractmodel import Model
from Detector.Utility.PydanticObject import DataObject

SET_TYPE = Tuple[np.ndarray, np.ndarray]
DATA_TYPE = list[SET_TYPE, SET_TYPE, Optional[SET_TYPE]]
INVERSE_INPUT_TYPE = Union[Tuple[np.ndarray, np.ndarray], np.ndarray]


def rescale_data(scaler: MinMaxScaler, data: Union[np.ndarray]) -> np.ndarray:
    """ Scale data back to real values

    Args:
        scaler: Scaler used to transform data
        data: data to rescale with the scaler

    Returns:
        rescaled data
    """
    if isinstance(data, tuple) or isinstance(data, list):
        data = data[1]
    if data.ndim > 2:
        for i in range(data.shape[1]):
            data_slice = data[:, i, :]
            data[:, i, :] = scaler.inverse_transform(data_slice)
    else:
        data = scaler.inverse_transform(data)
    return data


def prepare_data(model: Model, df: pd.DataFrame, data_object: DataObject, val_set: bool, test_set: bool) \
        -> Tuple[dict, dict, DATA_TYPE]:
    """ Prepare the data for training


    Args:
        model: model used
        df: data frame to use for training
        data_object: data_object
        val_set: make validation set
        test_set: make test set

    Returns:
        column of the train futures, column of the target, input_data [training, test, optional[validation]], training_df, test_df
    """
    # Process df
    if isinstance(data_object.target_col, list):
        target_index = {}
        for col in data_object.target_col:
            target_index[col] = df.columns.get_loc(col)
    else:
        target_index = {data_object.target_col: df.columns.get_loc(data_object.target_col)}

    train_index = {}
    for train_i in data_object.train_features:
        train_index[train_i] = df.columns.get_loc(train_i)

    df_numpy = df.to_numpy()

    input_data = model.get_data(df_numpy, train_index, target_index, val_set, test_set)
    return train_index, target_index, input_data


def split_series(data: np.ndarray, n_past: int, n_future: int, train_index: dict, target_index: dict) -> Tuple[
    np.ndarray, np.ndarray]:
    """ Split timeserie into X and y, because keras generator can't use multistep

    Args:
        data: Time series data
        n_past: no. of past observations
        n_future: no. of future observations
        train_index: in x but not in y
        target_index: target, not in x but is in y

    Returns:
        X, y
    """
    X, y = [], []
    for window_start in range(len(data)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(data):
            break
        # slicing the past and future parts of the window
        past, future = data[window_start:past_end, :], data[past_end:future_end, :]
        past = np.delete(past, list(target_index.values()), axis=1)
        future = np.delete(future, list(train_index.values()), axis=1)
        X.append(past)
        y.append(future)
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    return X, y


def inverse_scale(scaler: MinMaxScaler, train: INVERSE_INPUT_TYPE,
                  test: INVERSE_INPUT_TYPE, prediction: INVERSE_INPUT_TYPE) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Scale data back to original

    Args:
        scaler: scaler used to transform data
        train: training data
        test: testing data
        prediction: prediction

    Returns:
        rescaled train, test, prediction
    """
    train = rescale_data(scaler, train)
    test = rescale_data(scaler, test)
    prediction = rescale_data(scaler, prediction)
    return train, test, prediction
