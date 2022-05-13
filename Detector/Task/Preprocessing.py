# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Preprocess the raw input data

# Imports
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from Detector.Utility.Data_preprocessing.Cleansing import remove_flatliners
from Detector.Utility.Data_preprocessing.Transformation import resample, add_diastolic_systolic_bp
from Detector.Utility.Data_preprocessing.extract_info import get_x_values, get_y_values, get_full_curve, get_indices, \
    extract_values, make_datasets
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Task.preprocessing.Preprocessor_creator import PreprocessorCreator
from Detector.Utility.Util import nan_helper, get_markers

o = 1
# Variables
plot_resample_vs_decimate = False
time = 60
baseline_length = 40
future = 150
standing_length = 150
seconds = 1


def preprocessing(info_object: InfoObject) -> dict:
    """ Perform preprocessing

    Args:
        info_object: Object containing all needed information (set in Config.json)

    Returns:
         Dataframe, Markers, tags
    """
    # Make a logger
    logger = logging.getLogger(__name__)

    # Get the preprocessor we need for our current dataset
    Preprocessor = PreprocessorCreator.get_preprocessor(info_object.dataset, logger=logger, info_object=info_object)

    # Lists
    infs = []
    parameters = []
    x_dataframes = []
    x_oxy_dxy = []
    y_curves = []

    # get all subject files
    for subject in os.listdir(Path(info_object.file_loc)):
        challenges = {}
        markers = {}
        subject_path = Path(os.path.join(info_object.file_loc, subject))
        print(subject)
        for challenge_file in os.listdir(subject_path):
            mat_file = Path(os.path.join(subject_path, challenge_file))
            challenge = Path(challenge_file).stem.split("_")[-1]
            try:
                # preprocess our data with the right Preprocessor
                df, markers_dict, datadict = Preprocessor.get_df(mat_file)
                data_object = DataObject(**datadict)

                # interpolate nan
                if df.isnull().values.any():
                    logger.info("Interpolating NANs")
                    nans, x = nan_helper(np.array(df[data_object.target_col]))
                    df.loc[nans, data_object.target_col] = np.interp(x(nans), x(~nans),
                                                                     df[data_object.target_col][~nans])

                # Remove flatliners from data
                df = remove_flatliners(df, data_object)

                # If we want to resample, we resample with a value given for time_row in de config
                if int(1000 / data_object.hz) != info_object.time_row:
                    logger.info("resampled")
                    df, data_object = resample(df, data_object, info_object)

                # add diastolic and systolic values
                df, data_object = add_diastolic_systolic_bp(df, data_object, info_object)

                # Remove missing values
                df.dropna(inplace=True)

                challenges[challenge] = df

                # get the protocol markers standing or sitting
                df, stand_markers = get_markers(df, markers_dict)

                markers[challenge] = stand_markers

                tags = Preprocessor.get_tags(mat_file)
                # Save extra info for the Subject
                info = {"Data": challenges, "Markers": markers, "info": tags.sample}

                x_dataframes, x_oxy_dxy, y_curves, infs, parameters = make_datasets(data_object,
                                                                                    subject,
                                                                                    info,
                                                                                    baseline_length,
                                                                                    standing_length,
                                                                                    time,
                                                                                    future,
                                                                                    seconds,
                                                                                    (x_dataframes,
                                                                                     x_oxy_dxy,
                                                                                     y_curves,
                                                                                     infs,
                                                                                     parameters)
                                                                                    )

            except ValueError as e:  # Value
                print(f"{subject}:{challenge}: contains invalid data")
                print(e)
                continue

    dd = defaultdict(list)

    for d in parameters:  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)

    # merge all different dictionaries to one dataframe
    parameters_dataset = pd.DataFrame(dd)

    dd = defaultdict(list)

    for d in infs:  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)

    # merge all different dictionaries to one dataframe
    info_dataset = pd.DataFrame(dd)

    X = np.array(x_dataframes)
    full_curve = np.array(y_curves)
    return X, info_dataset, parameters_dataset, full_curve
