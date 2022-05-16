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
from statistics import mean

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# Variables
from Detector.Utility.Data_preprocessing.Transformation import scale3d, scale2d, reverse_scale2d, reverse_scale3d
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting.plotting import plot_comparison, plot_prediction
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

    X_unscaled = X
    X, x_scalers = scale3d(X_unscaled.copy(), data_object)

    if info_object.parameter_model:
        output_unscaled = np.array(parameters_values)
        output, out_scaler = scale2d(output_unscaled.copy(), data_object)
        out_scalers = None
    else:
        output_unscaled = full_curve
        output, out_scalers = scale3d(output_unscaled.copy(), data_object)
        out_scaler = None

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
            print(info_object.model)
            model_copy = ModelCreator.create_model(info_object.model, data_object=data_object,
                                                   input_shape=X.shape[1:],
                                                   output_shape=output.shape[1:],
                                                   gpu=use_gpu, plot_layers=True,
                                                   parameters=parameters)
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

        # Scale back the prediction
        if out_scalers is None and out_scaler is not None:
            prediction = reverse_scale2d(prediction, out_scaler)
        else:
            prediction = reverse_scale3d(prediction, out_scalers)

        if info_object.parameter_model:
            plot_comparison(info_object.model, info_dataset.iloc[test_indexes], list(parameters_values.columns),
                            prediction, output_unscaled[test_indexes], folder_name="figure/")
        else:
            for i in range(prediction.shape[0]):
                for target_index, target_name in enumerate(data_object.target_col):

                    sample = test_indexes[i]
                    information = info_dataset.iloc[sample]
                    path = f"{information.ID}_{information.challenge}_{int(information['repeat'])}"
                    if std is not None:
                        std_target = std[i]
                    else:
                        std_target = None
                    plot_prediction(
                            target_name, target_index, prediction[i], output[sample], std=std_target,
                        title= f"{target_name} {path}".replace('_', ' '), folder_name=path)
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
