# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Optimizing task, optimize a model with optuna and save to MLflow

# Imports
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from Detector.Utility.PydanticObject import DataObject, InfoObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.optimizer import Optimizer
from Detector.enums import Parameters


def optimize_model(x: np.ndarray,
                   parameters_values: pd.DataFrame, full_curve: np.ndarray, info_dataset: pd.DataFrame,
                   data_object: DataObject, info_object: InfoObject, fit_indexes: list):
    """ Optimizing a model with optuna, save the best parameters to MLflow

    Args:
        x: Dataframe containing input to use for the model
        parameters_values: Dataframe containing output to use for the model when predicting parameters
        full_curve: Dataframe containing output to use for the model when predicting full curve
        info_dataset: Dataframe containing information about the subject
        data_object: information retrieved from the data
        info_object: information from config
        fit_indexes: indexes to use for fitting
    """
    storage = f"study_{info_object.model}.db"

    # get tags
    keys_to_extract = ["index_col", "nirs_col", "target_col", "hz"]
    do = data_object.dict()
    tags = {key: do[key] for key in keys_to_extract}

    serializer = MLflowSerializer(nirs_data=info_object.nirs_input,
                                  dataset_name=info_object.dataset,
                                  parameter_expiriment=info_object.parameter_model,
                                  sample_tags=tags)
    old_run = check_for_optimized_run(serializer, name=info_object.model, storage_file=storage)

    # Check which output we want
    if info_object.parameter_model:
        output = parameters_values.iloc[fit_indexes]
    else:
        output = full_curve[fit_indexes]

    with mlflow.start_run(experiment_id=serializer.experiment_id, run_name=info_object.model):
        # Create Optuna optimizer
        optimizer = Optimizer(x[fit_indexes],
                              output,
                              info_dataset.iloc[fit_indexes],
                              info_object, data_object)
        # Perform optuna studies, and get the optuna study
        study = optimizer.optimize_parameters(storage=storage)
        # get the best trail
        trial = study.best_trial

        # write to mlflow
        mlflow.set_tag("phase", "Optimizing")
        mlflow.log_params(trial.params)
        mlflow.log_params(trial.system_attrs)
        mlflow.log_params(trial.user_attrs)
        mlflow.log_metric(Parameters.loss.value, trial.value)
        mlflow.log_metric("trials", len(study.trials))
        mlflow.log_artifact(storage, "study")
        # clean up old files
        os.remove(Path(storage))
        if old_run is not None:
            shutil.rmtree(Path(old_run))


def check_for_optimized_run(serializer: MLflowSerializer, name: str, storage_file: str) -> Optional[str]:
    """ Get the last optimized run and copy storage file out of MLflow

    Args:
        serializer: the MLflow serializer
        name: name used to get the study and run, is the model name
        storage_file: The file where the study will be saved to (is temp)

    Returns:
        if an old optimize run exists, returns the path of that run
    """
    # get the last optimization run
    last_run = serializer.get_last_optimized_run(name=name)
    # if there is no run we start clean
    if last_run is None:
        return
    # get the artifact store location
    artifact_uri = last_run.artifact_uri
    # remove uri suffix
    artifact_uri = artifact_uri.removeprefix(serializer.uri_start)
    # get the run path
    run_path = artifact_uri.removesuffix("artifacts")
    # get the study file
    artifact_uri = os.path.join(artifact_uri, f"study/{storage_file}")
    # copy to pwd for easier use
    try:
        shutil.copyfile(Path(artifact_uri), storage_file)
    except FileNotFoundError:
        logger = logging.getLogger(__name__)
        logger.warning("study file not found in existing study!")
        return
    return run_path
