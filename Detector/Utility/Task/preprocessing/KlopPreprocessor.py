# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Preprocessor for Klop dataset

# Imports
import os
from os.path import basename
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

from Detector.Utility.PydanticObject import TagsObject
from Detector.Utility.Task.preprocessing.PreprocessingFunctions import fetch_matlab_struct_data, add_value
from Detector.Utility.Task.preprocessing.abstractPreprocessor import Preprocessor
from Detector.enums import Files

MERGE_L_R = True


class KlopPreprocessor(Preprocessor):
    """ Perform preprocessing for Klop dataset """

    def __init__(self, logger, info_object):
        self.file_type = None
        self.logger = logger
        self.info_object = info_object

    def get_df(self, file) -> Tuple[pd.DataFrame, dict, dict]:
        # check type of file preprocessed or markerschecked
        file_name = basename(file)

        target_col = ["BP"]
        features = [
            "oxyvals",
            "dxyvals",
            "ADvalues",  # contains the movement sensor
        ]

        mat = loadmat(file)
        python_dict = fetch_matlab_struct_data(mat)

        # variables
        if "markerschecked" in file_name.lower():
            self.file_type = "markerschecked"
            python_dict, markers, index_col, index, hz, ad_v, ad_l = self._markerschecked_variables(python_dict)
        elif "preprocessed" in file_name.lower():
            self.file_type = "preprocessed"
            python_dict, markers, index_col, index, hz, ad_v, ad_l = self._prepocessed_variables(python_dict)
        else:
            raise ValueError("File name not as expected")

        for key in python_dict:
            if ad_v.lower() in key.lower():
                ad_v = key
            elif ad_l.lower() in key.lower():
                ad_l = key

        index_length = index.shape[0]
        features = [index_col] + target_col + features

        data = {}
        cols = []
        nirs_col = []
        number_of_rows = None
        for key, value in python_dict.items():
            if key in features:
                # If the value is not in the same frequency we decimate it
                if index_length not in value.shape:
                    value = decimate(x=value, q=2, ftype="fir")
                    value = value[:index_length]
                    if number_of_rows is None:
                        number_of_rows = value.shape[0]
                    elif number_of_rows != value.shape[0]:
                        self.logger.warning("data contains to many different lengths (2+)")
                        raise AssertionError("data contains to many different lengths")
                # if the value contains multiple dimensions we need to add them one by one
                elif value.ndim >= 2:
                    if key == ad_v:
                        columns = []
                        for col in python_dict[ad_l]:
                            name = col[0]
                            if "ACC" in name:
                                columns.append(name)
                    elif (key == "oxyvals") or (key == "dxyvals"):
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
                        nirs_col.append(f"{key}_long")
                        cols.append(f"{key}_long")
                        continue
                    else:
                        columns = [f"{key}_{i + 1}" for i in range(value.shape[1])]
                    cols.extend(columns)
                    data = add_value(data, columns, value)
                    continue
                cols.append(key)
                data[key] = value

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

        # convert acc columns to 1 or 2 columns
        acc_df = df.filter(like="ACC")
        drop_cols = list(acc_df.columns)
        Z = acc_df.filter(regex="Z").mean(axis=1)
        df.drop(drop_cols, axis=1, inplace=True)
        df["Z_movement"] = Z

        dataframe_cols = list(df.columns)
        [dataframe_cols.remove(col) for col in target_col]

        datadict = {
            "index_col": index_col,
            "target_col": target_col,
            "nirs_col": nirs_col,
            "movement_features": ["Z_movement"],
            "features": dataframe_cols,
            "hz": hz,
        }

        return df, markers, datadict

    @staticmethod
    def _markerschecked_variables(python_dict: dict) -> Tuple[dict, dict, str, np.ndarray, int, str, str]:
        stages: list = [
            "start1",
            "move1",
            "stand1",
            "stop1",
            "start2",
            "move2",
            "stand2",
            "stop2",
            "start3",
            "move3",
            "stand3",
            "stop3"
        ]
        ad_v = "ADvalues"
        ad_l = "ADlabel"
        markers = "markerstijd"
        marker_times = python_dict[markers]
        marker_dict = {}
        for i, stage in enumerate(stages):
            marker_dict[stage] = marker_times[i]
        index_col = "nirs_time"
        # get HZ
        hz_col = "fs_nirs"
        hz = int(python_dict[hz_col])
        # get index values
        index: np.ndarray = python_dict[index_col]
        index = index - index[0]
        python_dict.pop(index_col)
        return python_dict, marker_dict, index_col, index, hz, ad_v, ad_l

    @staticmethod
    def _prepocessed_variables(python_dict: dict) -> Tuple[dict, dict, str, np.ndarray, int, str, str]:
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

        ad_v = "ADvalues"
        ad_l = "ADlabel"
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

        return python_dict, marker_dict, index_col, index, hz, ad_v, ad_l

    def get_tags(self, file) -> TagsObject:
        tags = {"sample": {}, "tags": {}}
        filename = os.path.basename(file)

        if self.file_type == "preprocessed":
            _, SampleID, challenge = filename.split("_")
        elif self.file_type == "markerschecked":
            SampleID, challenge, _ = filename.split("_")
        else:
            raise ValueError("file not recognised")

        # get extra information on Sample
        sample_data = pd.read_csv(Files.tags.value, index_col=0)
        sample = sample_data.loc[SampleID]
        tags["ID"] = SampleID
        tags["sample"]["Age"] = sample.loc["Age"]
        tags["sample"]["Sex"] = sample.loc["Sex"]

        # get information about the data
        tags = TagsObject(**tags)
        return tags
