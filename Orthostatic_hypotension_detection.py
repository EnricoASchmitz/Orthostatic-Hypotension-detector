# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script:
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import streamlit as st

from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from util_app import InfoObject, get_df, get_clicks, get_markers, get_dataset, scale3d, make_curves, plot_curve, \
    hampel_filter

Click = 0


def load_json():
    st.write("# Parameter models")
    st.write("## Information")
    config_file = st.file_uploader("pick a .json file", type="json")
    preproccesor = st.radio("Pick a preprocessor to use", ('klop',))
    clicks = get_clicks(st.button("Click me"))
    return config_file, preproccesor, clicks


def load_input():
    # Get the preprocessor we need for our current dataset
    # Make a logger
    logger = logging.getLogger(__name__)
    input_file = st.file_uploader("pick a .mat file", type="mat")
    return logger, input_file


def load_names():
    oxy_name = st.text_input('Oxy channel name', "oxyvals")
    dxy_name = st.text_input('Dxy channel name', "dxyvals")
    return oxy_name, dxy_name


def load_data():
    df, markers, hz = get_df(input_file, oxy_name, dxy_name)
    with st.spinner('Performing Hampel filter...'):
        for col in df.columns:
            df[col] = hampel_filter(df[col], window_size=hz)
    # get the protocol markers standing or sitting
    with st.spinner('Getting Markers...'):
        stand_markers = get_markers(df, markers)
    with st.spinner('Extracting NIRS...'):
        x_oxy_dxy = get_dataset(df, stand_markers, logger)
        X = np.array(x_oxy_dxy)
    return X


def load_scalers():
    scaler_in_file = st.file_uploader("pick a .save file, input scaler", type="save")
    scaler_out_file = st.file_uploader("pick a .save file, output scaler", type="save")
    if scaler_in_file is None or scaler_out_file is None :
        st.warning('Missing scaler files!')
        st.stop()
    return joblib.load(scaler_in_file), joblib.load(scaler_out_file)


def load_model(input_shape, output_shape):
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
    artifact_uri_with_lead = best_run.artifact_uri
    # remove uri suffix
    artifact_uri = artifact_uri_with_lead.removeprefix(serializer.uri_start)
    if artifact_uri_with_lead == artifact_uri:
        serializer.uri_start = serializer.uri_start[:-1]
        artifact_uri = artifact_uri_with_lead.removeprefix(serializer.uri_start)
    # get the model folder
    model_folder = Path(os.path.join(artifact_uri, f"model/"))
    model = ModelCreator.create_model(model_name,
                                      input_shape=input_shape,
                                      output_shape=output_shape,
                                      gpu=True, plot_layers=True,
                                      parameters=model_parameters)
    # load fitted model
    try:
        model.load_model(model_folder)
    except OSError:
        # remove leading "/" or "\" and try again
        model_folder = Path(str(model_folder).lstrip("/\\"))
        model.load_model(model_folder)
    return model


if __name__ == "__main__":
    config_file, preproccesor, clicks = load_json()
    if clicks >= 1:
        if config_file is None:
            st.warning('No config file given!')
            st.stop()
        logger, input_file = load_input()
        if clicks >= 2:
            if input_file is None:
                st.warning('No input file given!')
                st.stop()
            oxy_name, dxy_name = load_names()
            if clicks >= 3:
                scalers_in, scaler_out = load_scalers()
                output_shape = len(scalers_in), scalers_in[0].n_features_in_
                if clicks >= 4:

                    X = load_data()
                    model = load_model(input_shape=X.shape[1:], output_shape=output_shape)
                    X_scaled = scale3d(X, scalers_in)
                    st.write("## Prediction")
                    with st.spinner('Making prediction...'):
                        prediction = model.predict(X_scaled)
                        # Scale back the prediction
                        prediction_array = scaler_out.inverse_transform(prediction)
                        cols = ['BP_systolic_drop', 'BP_systolic_drop_index', 'BP_systolic_drop_rate',
                                'BP_systolic_recovery_15', 'BP_systolic_recovery_20',
                                'BP_systolic_recovery_30', 'BP_systolic_recovery_40',
                                'BP_systolic_recovery_50', 'BP_systolic_recovery_60',
                                'BP_systolic_recovery_120', 'BP_systolic_recovery_150',
                                'BP_systolic_drop_per_sec', 'BP_systolic_recovery_rate_60',
                                'BP_diastolic_drop', 'BP_diastolic_drop_index',
                                'BP_diastolic_drop_rate', 'BP_diastolic_recovery_15',
                                'BP_diastolic_recovery_20', 'BP_diastolic_recovery_30',
                                'BP_diastolic_recovery_40', 'BP_diastolic_recovery_50',
                                'BP_diastolic_recovery_60', 'BP_diastolic_recovery_120',
                                'BP_diastolic_recovery_150', 'BP_diastolic_drop_per_sec',
                                'BP_diastolic_recovery_rate_60']
                        reconstruct_params = ['BP_systolic_drop', 'BP_systolic_drop_index', 'BP_systolic_recovery_15',
                                              'BP_systolic_recovery_20', 'BP_systolic_recovery_30',
                                              'BP_systolic_recovery_40', 'BP_systolic_recovery_50',
                                              'BP_systolic_recovery_60', 'BP_systolic_recovery_120',
                                              'BP_systolic_recovery_150', 'BP_diastolic_drop',
                                              'BP_diastolic_drop_index', 'BP_diastolic_recovery_15',
                                              'BP_diastolic_recovery_20', 'BP_diastolic_recovery_30',
                                              'BP_diastolic_recovery_40', 'BP_diastolic_recovery_50',
                                              'BP_diastolic_recovery_60', 'BP_diastolic_recovery_120',
                                              'BP_diastolic_recovery_150']
                        recovery_times = [15, 20, 30, 40, 50, 60, 120, 150]

                        prediction = pd.DataFrame(prediction_array, columns=cols).copy()
                        pred_curve = make_curves(prediction, reconstruct_params, recovery_times)
                        for i in range(pred_curve.shape[0]):
                            st.write(f"Prediction for repetition {i}")
                            for target_index, target_name in enumerate(["SBP", "DBP"]):
                                fig = plot_curve(i, pred_curve, target_index, target_name)
                                st.plotly_chart(fig, use_container_width=True)
                    get_clicks(False, remove=True)

    if st.button("Reset"):
        get_clicks(False, remove=True)
