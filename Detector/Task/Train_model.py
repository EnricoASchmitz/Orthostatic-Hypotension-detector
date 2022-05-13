# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Train a model and write to MLflow

# Imports
import fnmatch
import gc
import logging
import os
from collections import defaultdict
from copy import deepcopy
from statistics import mean

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split, KFold

# Variables
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting.plotting import plot_comparison
from Detector.Utility.PydanticObject import DataObject, InfoObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.model_functions import check_gpu, fit_and_predict, predicting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(X: np.ndarray, info_dataset: pd.DataFrame,
                parameters_values: pd.DataFrame, full_curve: np.ndarray,
                data_object: DataObject, info_object: InfoObject):
    """ Train a model, save to MLflow

    Args:
        X: Dataframe containing input to use for the model
        info_dataset: Dataframe containing information about the subjects
        parameters_values: Dataframe containing output to use for the model when predicting parameters
        full_curve: Dataframe containing output to use for the model when predicting full curve
        data_object: information retrieved from the data
        info_object: information from config
    """
    use_gpu = check_gpu()

    logger = logging.getLogger(__name__)

    serializer = MLflowSerializer(dataset_name=info_object.dataset, data_object=data_object,
                                  parameter_expiriment=info_object.parameter_model, sample_tags={})
    last_optimized_run = serializer.get_last_optimized_run(info_object.model)
    if last_optimized_run is not None:
        run = mlflow.get_run(last_optimized_run.run_id)
        parameters = run.data.params
    else:
        parameters = None
    logger.info(f"creating model {info_object.model}")
    model = ModelCreator.create_model(info_object.model, data_object=data_object,
                                      input_shape=(0, 0, 0),
                                      output_shape=(0, 0, 0),
                                      gpu=use_gpu, plot_layers=True,
                                      parameters=parameters)

    if info_object.parameter_model:
        output = np.array(parameters_values)
    else:
        output = full_curve

    with mlflow.start_run(experiment_id=serializer.experiment_id, run_name=info_object.model):
        mlflow.set_tag("phase", "training")

        # cross val
        step = 0
        n_splits = 5
        tscv = KFold(n_splits=n_splits)

        loss_dicts = []
        models_list = []

        fit_indexes, test_indexes = train_test_split(range(len(X)))

        for indexes in tscv.split(fit_indexes):
            logger.warning(f"start cv: {step}")
            # collect
            gc.collect()
            model_copy = deepcopy(model)
            model_copy, loss_values = fit_and_predict(info_object=info_object,
                                                      logger=logger,
                                                      input_values=X,
                                                      output_values=output,
                                                      model=model_copy,
                                                      step=step,
                                                      indexes=indexes)
            step += 1
            loss_dicts.append(loss_values)
            models_list.append(model_copy)

        # Get best model
        mae_loss = []
        for k in loss_dicts:
            mae_loss.append(k['mae'])
        best_k = np.array(mae_loss).argmin()

        model = models_list[best_k]

        # Mlflow log loss values
        losses = defaultdict(list)
        for d in loss_dicts:  # you can list as many input dicts as you want here
            for key, value in d.items():
                losses[key].append(value)
        avg_loss = {}
        for loss, vals in losses.items():
            avg_loss[f"avg_{loss}"] = mean(vals)

        mlflow.log_metrics(avg_loss)

        prediction, std, time = predicting(model, logger, X[test_indexes])

        plot_comparison(info_object.model, info_dataset.iloc[test_indexes], list(parameters_values.columns), prediction, output[test_indexes], folder_name="figure/")

        # save parameters to mlflow
        mlflow.log_params(model.get_parameters())

        # save model
        model.save_model()

        # remove file, since it is saved in MLflow
        for file in os.listdir('.'):
            if (fnmatch.fnmatch(file, 'model.png')) or (fnmatch.fnmatch(file, 'mapper.png')):
                mlflow.log_artifact(file, "figure")
                os.remove(file)
            elif (fnmatch.fnmatch(file, 'model.*')) or (fnmatch.fnmatch(file, 'mapper.h5')):
                mlflow.log_artifact(file, "model")
                os.remove(file)

        logger.info(f"Done {info_object.model}")
