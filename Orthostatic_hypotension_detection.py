# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script:
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from Detector.Utility.Data_preprocessing.Cleansing import remove_flatliners, remove_unrealistic_values, hampel_filter
from Detector.Utility.Data_preprocessing.Transformation import resample, add_diastolic_systolic_bp, scale3d, \
    reverse_scale2d, scale2d
from Detector.Utility.Data_preprocessing.extract_info import make_datasets, make_curves
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting.plotting import plot_curves
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.preprocessing.Preprocessor_creator import PreprocessorCreator
from Detector.Utility.Util import nan_helper, get_markers


def preprocess_data(input_file):
    # Lists
    infs = []
    parameters = []
    x_dataframes = []
    x_oxy_dxy = []
    y_curves = []
    # preprocess data
    df, markers_dict, datadict = Preprocessor.get_df(input_file, file_name=input_file.name)
    data_object = DataObject(**datadict)
    data_object.scaler = MinMaxScaler(feature_range=(-1, 1))
    # interpolate nan
    if df.isnull().values.any():
        logger.info("Interpolating NANs")
        nans, x = nan_helper(np.array(df[data_object.target_col]))
        df.loc[nans, data_object.target_col] = np.interp(x(nans), x(~nans),
                                                         df[data_object.target_col][~nans])

    # perform Hampel filter to remove outliers with a window of 1 sec
    for col in data_object.target_col:
        df[col] = hampel_filter(df[col], window_size=data_object.hz)
    for col in data_object.nirs_col:
        df[col] = hampel_filter(df[col], window_size=data_object.hz)
    # Remove values above and below normal
    df = remove_unrealistic_values(df, data_object)
    # Remove flatliner artifacts
    df = remove_flatliners(df, data_object)

    # add diastolic and systolic values
    df, data_object = add_diastolic_systolic_bp(df, data_object, info_object)

    # If we want to resample, we resample with a value given for time_row in de config
    if int(1000 / data_object.hz) != info_object.time_row:
        logger.info("resampled")
        df, data_object = resample(df, data_object, info_object)
    # get the protocol markers standing or sitting
    stand_markers = get_markers(df, markers_dict)

    # Save extra info for the Subject
    info = {"Data": df,
            "Markers": stand_markers,
            "Challenge": "test",
            "info": {}
            }
    # Get the data
    x_dataframes, x_oxy_dxy, y_curves, infs, parameters = make_datasets(data_object,
                                                                        "test",
                                                                        info,
                                                                        1,
                                                                        (x_dataframes,
                                                                         x_oxy_dxy,
                                                                         y_curves,
                                                                         infs,
                                                                         parameters)
                                                                        )
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

    if info_object.nirs_input:
        X = np.array(x_oxy_dxy)
    else:
        X = np.array(x_dataframes)
    full_curve = np.array(y_curves)
    x, x_scalers = scale3d(X.copy(), data_object)
    output_unscaled = np.array(parameters_dataset)
    output, out_scaler = scale2d(output_unscaled.copy(), data_object)
    return x, output, parameters_dataset, full_curve, data_object, out_scaler


if __name__ == "__main__":
    st.write("# Parameter models")
    st.write("## Information")
    config_file = st.file_uploader("pick a .json file", type="json")
    preproccesor = st.radio("Pick a preprocessor to use", ('klop',))

    info_object = None
    if config_file is not None:
        info_dict = json.load(config_file)[str(preproccesor)]
        info_object = InfoObject(**info_dict)
        # Get the preprocessor we need for our current dataset
        # Make a logger
        logger = logging.getLogger(__name__)
        Preprocessor = PreprocessorCreator.get_preprocessor(info_object.dataset, logger=logger, info_object=info_object)
        input_file = st.file_uploader("pick a .mat file", type="mat")

        if input_file is not None:
            serializer = MLflowSerializer(nirs_data=True,
                                          dataset_name=preproccesor,
                                          parameter_expiriment=True,
                                          sample_tags={})
            best_run = serializer.get_best_model()

            # Get parameters
            model_name = best_run["tags.mlflow.runName"]
            run = mlflow.get_run(best_run.run_id)

            model_parameters = run.data.params
            # get the artifact store location
            artifact_uri = best_run.artifact_uri
            # remove uri suffix
            artifact_uri = artifact_uri.removeprefix(serializer.uri_start)
            # get the model folder
            model_folder = Path(os.path.join(artifact_uri, f"model/"))

            with st.spinner('Preprocessing data...'):
                x, output, parameters_dataset, full_curve, data_object, out_scaler = preprocess_data(input_file)

            model = ModelCreator.create_model(model_name,
                                              input_shape=x.shape[1:],
                                              output_shape=output.shape[1:],
                                              gpu=True, plot_layers=True,
                                              parameters=model_parameters)
            # load fitted model
            try:
                model.load_model(model_folder)
            except OSError:
                # remove leading "/" or "\" and try again
                model_folder = Path(str(model_folder).lstrip("/\\"))
                model.load_model(model_folder)

            st.write("Prediction")
            with st.spinner('Making prediction...'):
                prediction = model.predict(x)
            # Scale back the prediction
            prediction_array = reverse_scale2d(prediction, out_scaler)
            prediction = pd.DataFrame(prediction_array, columns=parameters_dataset.columns).copy()
            true_curve, pred_curve = make_curves(prediction, parameters_dataset, data_object.reconstruct_params,
                                                 data_object.recovery_times, list(parameters_dataset.index))
            st.dataframe(prediction)
            for i in range(pred_curve.shape[0]):
                st.write(f"Prediction for repetition {i}")
                for target_index, target_name in enumerate(data_object.target_col):
                    plot_index = list(parameters_dataset.index)[i]
                    sample = full_curve[plot_index]
                    fig = plot_curves(sample, i, pred_curve, true_curve, target_index, target_name, streamlit_bool=True)
                    st.plotly_chart(fig, use_container_width=True)
