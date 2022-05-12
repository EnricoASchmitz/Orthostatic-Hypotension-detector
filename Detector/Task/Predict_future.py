# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting.plotting import plot_prediction
from Detector.Utility.PydanticObject import DataObject, TagsObject, InfoObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.model_functions import check_gpu
from Detector.Utility.Task.setup_data import prepare_data, inverse_scale
from Detector.enums import Parameters


def predict_future(df: pd.DataFrame, n_in_steps, n_out_steps, n_features, info_object: InfoObject,
                   data_object: DataObject, tags: TagsObject, scaler):
    use_gpu = check_gpu()
    logger = logging.getLogger(__name__)
    serializer = MLflowSerializer(dataset_name=info_object.dataset, data_object=data_object, sample_tags=tags.sample,
                                  n_in_steps=n_in_steps, n_out_steps=n_out_steps)
    last_run = serializer.get_last_training_run(info_object.model)
    if last_run is None:
        raise AssertionError("First fit a model before making predictions")

    # Get parameters
    run = mlflow.get_run(last_run.run_id)
    parameters = run.data.params
    # get the artifact store location
    artifact_uri = last_run.artifact_uri
    # remove uri suffix
    artifact_uri = artifact_uri.removeprefix(serializer.uri_start)
    # get the model folder
    model_folder = Path(os.path.join(artifact_uri, f"model/"))

    # Get the number of features by removing number of movement and target columns
    n_in_features = n_features - len(data_object.movement_features) - len(data_object.target_col)

    # create a model
    n_mov_features = len(data_object.movement_features)
    model = ModelCreator.create_model(info_object.model, data_object=data_object,
                                      n_in_steps=n_in_steps,
                                      n_out_steps=n_out_steps,
                                      n_in_features=n_in_features, n_mov_features=n_mov_features, gpu=use_gpu,
                                      parameters=parameters)
    # load fitted model
    try:
        model.load_model(model_folder)
    except OSError:
        # remove leading "/" or "\" and try again
        model_folder = Path(str(model_folder).lstrip("/\\"))
        model.load_model(model_folder)

    # get data
    train_index, target_index, input_data = prepare_data(model,
                                                         df,
                                                         data_object,
                                                         val_set=False,
                                                         test_set=True)
    # train set used as first input
    train_set, test_set = input_data

    # check how much prediction we need to do
    num_chunks = int(np.ceil(Parameters.minutes.value * 60 * data_object.hz / n_out_steps))
    future_prediction, std = model.predict_future(train_set[0], num_chunks)
    # reverse data scaling
    test_set_out = test_set[1]
    y_out = test_set_out.reshape(-1, test_set_out.shape[-1])
    y = y_out[::n_out_steps, :]

    future_prediction = np.expand_dims(future_prediction, axis=1)
    y = np.expand_dims(y, axis=1)

    test_data = y[:len(future_prediction)]
    pred_data = future_prediction[:len(y)]

    with mlflow.start_run(experiment_id=serializer.experiment_id, run_id=last_run.run_id):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # plot architecture outputs
        keys = set(list(train_index.keys()) + list(target_index.keys()))
        remaining_keys = set(list(df.columns)) - keys
        for i, name in enumerate(remaining_keys):
            if std is not None:
                standard_dev = std[..., -1, i]
            else:
                standard_dev = None
            title = f"{name} future prediction)".replace("_", " ")
            plot_prediction(target_name=name, target_index=len(target_index) + i, train=train_set[1],
                            prediction=pred_data,
                            true=test_data, title=title, std=standard_dev, folder_name=f"prediction_{timestamp}")

        train, test, prediction = inverse_scale(scaler=scaler, train=train_set,
                                                test=test_data,
                                                prediction=pred_data)

        # plot prediction for targets
        for name, column in target_index.items():
            plot_prediction(target_name=name, target_index=column, train=train, prediction=prediction, true=test,
                            title=f"{name} future prediction".replace("_", " "), folder_name=f"prediction_{timestamp}")
