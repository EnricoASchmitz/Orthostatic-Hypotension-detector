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
from Detector.Utility.Data_preprocessing.extract_info import get_x_values, get_y_values, get_full_curve
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Task.preprocessing.Preprocessor_creator import PreprocessorCreator
from Detector.Utility.Util import nan_helper, get_markers

o = 1
# Variables
plot_resample_vs_decimate = False


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

    subjects = {}
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
            except ValueError as e:  # Value
                print(f"{subject}:{challenge}: contains invalid data")
                continue

        if challenges:
            tags = Preprocessor.get_tags(mat_file)
            # Save extra info for the Subject
            info = {
                "Data": challenges,
                "Markers": markers
            }

            info["info"] = tags.sample
            subjects[subject] = info

    # Now merge this into a dataframe containing all subjects
    # Variables
    time = 60
    future = 150
    past = 180
    seconds = 1

    # Lists
    infs = []
    parameters = []
    x_dataframes = []
    x_oxy_dxy = []
    y_curves = []

    # Loop over all subjects
    for sub, info in subjects.items():
        print(sub)
        dfs = info['Data'].copy()
        # Loop over all dataframes
        for chal, df in dfs.items():
            h_stages = df['stage']
            # Get protocol
            sitting = h_stages.str.contains("start", case=False)
            stand = h_stages.str.contains("stand", case=False)
            stop_index = df.index[0]
            if chal == "FSit" or chal == "FSup":
                pass
            else:
                break
            for repeat in range(0, 3):
                # Get index when the repeat starts
                start_index = sitting[stop_index:].idxmax()
                # Get index when we stand up during the repeat
                stand_index = stand[start_index:].idxmax()
                # Get index when this repeats ends
                stop_index = stand[stand_index:].idxmin()
                # Make sure that we have a stop and start index
                if stand_index == stop_index:
                    stop_index = stand.index[-1]
                if start_index == stand_index:
                    print(f"Missing repeat: {repeat + 1} with challenge:{chal}")
                    continue
                # Save the patient info
                par = info["info"]
                par["ID"] = sub

                # Dicts
                params = {}
                x = {}
                x_nirs = {}
                bp_dict = {}
                # get values for systolic and diastolic
                for BP_type in data_object.target_col:
                    # get x values for corresponding data
                    x = get_x_values(x, df[BP_type].loc[start_index:stand_index].copy(), BP_type)
                    # get parameters about the standing part
                    params = get_y_values(params, df[BP_type].copy(), time, stand_index, BP_type)
                    # Get the full BP during standing
                    bp_dict = get_full_curve(bp_dict, BP_type, df[BP_type].copy(), stand_index, stop_index, seconds)

                # get values for oxy and dxy
                for Nirs_type in data_object.nirs_col:
                    x_nirs = get_x_values(x_nirs, df[Nirs_type].loc[start_index:stand_index].copy(), Nirs_type)

                # Convert dictionary to dataframe and then to numpy array
                x_df = pd.DataFrame(x).interpolate()
                x_array = np.array(x_df)[-past:]

                x_nirs_df = pd.DataFrame(x_nirs).interpolate()
                x_nirs_array = np.array(x_nirs_df)[-past:]

                # calculate expected rows in the daframe
                future_steps = int(future / seconds)

                y_curve = pd.DataFrame(bp_dict).interpolate()
                y_curve_array = np.array(y_curve)[:future_steps]

                # Make sure data is in the right format or we skip the repeat
                if x_array.shape[0] == past and y_curve_array.shape[0] == future_steps:
                    x_dataframes.append(x_array)
                    x_oxy_dxy.append(x_nirs_array)
                    y_curves.append(y_curve_array)
                    parameters.append(params)
                    infs.append(par)
                elif x_array.shape[0] == past and y_curve_array.shape[0] != future_steps:
                    print(
                        f"y curve not long enough; {y_curve_array.shape[0]}/{future_steps}; Subject {sub}; challenge: {chal}; repeat: {repeat + 1}")
                else:
                    print(f"X incorrect; Subject {sub}; challenge: {chal}; repeat: {repeat + 1}")

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
