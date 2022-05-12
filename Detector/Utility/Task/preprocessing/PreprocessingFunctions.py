# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions needed for preprocessing

# Imports
import json
from typing import Union, List, Optional

import numpy as np

from Detector.Utility.PydanticObject import InfoObject
from Detector.enums import Files


def fetch_matlab_struct_data(matlab, imported_data=None, key='data',
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
                raise KeyError('Invalid key = {0:s}.'.format(key))

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
                    indent = '  ' * level
                    child = matlab_void[field].squeeze()
                    if printing:
                        print(indent + '{0:s}: shape = {1}'.format(field, child.shape))
                    if child.shape == (1, 1) and isinstance(child[0, 0], np.void):
                        fetch_matlab_struct_data(child, imported_data, level=level + 1, parent_field=field)
                    else:
                        if parent_field is not None:
                            key = parent_field + ':' + field
                        else:
                            key = field
                        imported_data[key] = child
    return imported_data


def add_value(dictionary: dict, columns: Union[str, List[str]], value: np.array) -> dict:
    """ add a multidimensional data to the dictionary

    Args:
        dictionary: dictionary containing the data
        columns: columns to add
        value: values from the column

    Returns:
        dictionary: used to make 3+dim to dataframe
    """
    for A, B in zip(columns, value.transpose()):
        dictionary[A] = B
    return dictionary


def create_info_object(dataset: Optional[str] = None, info_dict: Optional[dict] = None):
    """ Create an info object

    Args:
        dataset: data set to use
        info_dict: dictionary to use for creation of object, if None use Config.json

    Raises:
        ValueError: if dataset and info_dict not specified.

    Returns:
        Object containing all the needed information
    """
    if (dataset is None) and (info_dict is None):
        raise ValueError("Missing a dataset for config or a dict containing the information")

    with open(Files.config.value, 'r') as f:
        json_dict = json.load(f)[str(dataset)]
        json_dict.update(info_dict)
        info_dict = json_dict.copy()
    return InfoObject(**info_dict)
