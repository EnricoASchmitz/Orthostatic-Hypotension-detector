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
from sklearn.model_selection import LeaveOneGroupOut

from Detector.Utility.Data_preprocessing.Transformation import scale3d, scale2d, reverse_scale2d
from Detector.Utility.Data_preprocessing.extract_info import make_curves
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting.plotting import plot_comparison, plot_curves, plot_bars
from Detector.Utility.PydanticObject import DataObject, InfoObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.model_functions import check_gpu, fit_and_predict, predicting
from Detector.enums import Parameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def train_model(x: np.ndarray, info_dataset: pd.DataFrame,
                parameters_values: pd.DataFrame, full_curve: np.ndarray,
                data_object: DataObject, info_object: InfoObject, fit_indexes: list, test_indexes: list):
    """ Train a model, save to MLflow

    Args:
        x: Dataframe containing input to use for the model
        info_dataset: Dataframe containing information about the subjects
        parameters_values: Dataframe containing output to use for the model when predicting parameters
        full_curve: Dataframe containing output to use for the model when predicting full curve
        data_object: information retrieved from the data
        info_object: information from config
        fit_indexes: indexes to use for fitting
        test_indexes: indexes to use for testing
    """
    use_gpu = check_gpu()

    logger = logging.getLogger(__name__)

    # get tags
    keys_to_extract = ["index_col", "nirs_col", "target_col", "hz"]
    do = data_object.dict()
    tags = {key: do[key] for key in keys_to_extract}

    serializer = MLflowSerializer(dataset_name=info_object.dataset,
                                  sample_tags=tags)
    last_optimized_run = serializer.get_last_optimized_run(info_object.model)
    if last_optimized_run is not None:
        run = mlflow.get_run(last_optimized_run.run_id)
        parameters = run.data.params
    else:
        parameters = None
    logger.info(f"creating model {info_object.model}")

    X_unscaled = x
    x, x_scalers = scale3d(X_unscaled.copy(), data_object)

    output_unscaled = np.array(parameters_values)
    output, out_scaler = scale2d(output_unscaled.copy(), data_object)

    with mlflow.start_run(experiment_id=serializer.experiment_id, run_name=info_object.model):
        mlflow.set_tag("phase", "training")

        # cross val
        step = 0
        ids = info_dataset.iloc[fit_indexes].ID
        logo = LeaveOneGroupOut()

        loss_dicts = []
        models_list = []

        logo.get_n_splits(groups=ids)

        for indexes in logo.split(fit_indexes, groups=ids):
            logger.info(f"start cv: {step}")
            # collect
            gc.collect()
            logger.info(info_object.model)
            model_copy = ModelCreator.create_model(info_object.model,
                                                   input_shape=x.shape[1:],
                                                   output_shape=output.shape[1:],
                                                   gpu=use_gpu, plot_layers=True,
                                                   parameters=parameters)
            model_copy, loss_values = fit_and_predict(info_object=info_object,
                                                      logger=logger,
                                                      input_values=x,
                                                      output_values=output,
                                                      model=model_copy,
                                                      step=step,
                                                      indexes=indexes,
                                                      scaler=out_scaler,
                                                      rescale_function=reverse_scale2d)
            step += 1
            loss_dicts.append(loss_values)
            models_list.append(model_copy)

        # Get best model
        list_loss = []
        for k in loss_dicts:
            list_loss.append(k[Parameters.loss.value])
        best_k = np.array(list_loss).argmin()

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

        prediction, std, time = predicting(model, logger, x[test_indexes])

        # Scale back the prediction
        prediction_array = reverse_scale2d(prediction, out_scaler)

        plot_comparison(info_object.model, info_dataset.iloc[test_indexes], list(parameters_values.columns),
                        prediction_array, output_unscaled[test_indexes], folder_name="figure/")
        prediction = pd.DataFrame(prediction_array, columns=parameters_values.columns).copy()
        true_curve, pred_curve = make_curves(prediction, parameters_values, data_object.reconstruct_params,
                                             data_object.recovery_times, test_indexes)
        for i in range(pred_curve.shape[0]):
            for target_index, target_name in enumerate(data_object.target_col):
                plot_index = test_indexes[i]
                sample = full_curve[plot_index]
                information = info_dataset.iloc[plot_index]
                path = f"figure/prediction/{information.ID}/{information.challenge}/{int(information['repeat'])}"
                plot_curves(sample, i, pred_curve, true_curve, target_index, target_name, folder_name=path)

        plot_bars(list(parameters_values.columns), output_unscaled[test_indexes], prediction_array,
                  output_unscaled[fit_indexes], folder_name="figure/bar_plots/")

        # save parameters to mlflow
        mlflow.log_params(model.get_parameters())

        # save model
        model.save_model()

        # remove file, since it is saved in MLflow
        for file in os.listdir("."):
            if fnmatch.fnmatch(file, "model.png"):
                mlflow.log_artifact(file, "figure")
                os.remove(file)
            elif fnmatch.fnmatch(file, "model.*"):
                mlflow.log_artifact(file, "model")
                os.remove(file)

        logger.info(f"Done {info_object.model}")
