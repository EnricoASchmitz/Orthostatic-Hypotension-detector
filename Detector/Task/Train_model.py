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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# Variables
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.PydanticObject import DataObject, InfoObject, TagsObject
from Detector.Utility.Serializer.Serializer import MLflowSerializer
from Detector.Utility.Task.model_functions import check_gpu, fit_and_predict
from Detector.Utility.Task.setup_data import prepare_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(df: pd.DataFrame, data_object: DataObject, info_object: InfoObject, scaler: MinMaxScaler,
                n_in_steps: int, n_out_steps: int, n_features: int,
                tags: TagsObject):
    """ Train a model, save to MLflow

    Args:
        df: Dataframe to use for the model
        data_object: information retrieved from the data
        info_object: information from config
        scaler: scaler used to transform data
        n_in_steps: number of input steps
        n_out_steps: number of output steps
        n_features: number of features
        tags: tags to add to mlflow
    """
    use_gpu = check_gpu()

    logger = logging.getLogger(__name__)

    serializer = MLflowSerializer(dataset_name=info_object.dataset, data_object=data_object, sample_tags=tags.sample,
                                  n_in_steps=n_in_steps, n_out_steps=n_out_steps)
    last_optimized_run = serializer.get_last_optimized_run(info_object.model)
    if last_optimized_run is not None:
        run = mlflow.get_run(last_optimized_run.run_id)
        parameters = run.data.params
    else:
        parameters = None
    logger.info(f"creating model {info_object.model}")
    # without BP and movement
    n_in_features = n_features - len(data_object.movement_features) - len(data_object.target_col)
    # movement
    n_mov_features = len(data_object.movement_features)
    model = ModelCreator.create_model(info_object.model, data_object=data_object,
                                      n_in_steps=n_in_steps,
                                      n_out_steps=n_out_steps, n_in_features=n_in_features,
                                      n_mov_features=n_mov_features, gpu=use_gpu, plot_layers=True,
                                      parameters=parameters)
    assert np.all((df.to_numpy() >= 0))
    train_index, target_index, input_data = prepare_data(model,
                                                         df,
                                                         data_object,
                                                         val_set=False,
                                                         test_set=True)

    # only use fitting data and save testing_data for predicting future
    fitting_data, testing_data = input_data
    with mlflow.start_run(experiment_id=serializer.experiment_id, run_name=info_object.model):
        mlflow.set_tag("phase", "training")
        mlflow.set_tag("input_steps", n_in_steps)
        mlflow.set_tag("output_steps", n_out_steps)
        mlflow.set_tag("n_features", n_features)
        mlflow.set_tags(tags.tags)

        # fit model
        movement_index = {}
        for movement_col in data_object.movement_features:
            movement_index[movement_col] = train_index[movement_col] - len(target_index)

        # cross val
        step = 0
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits)

        loss_dicts = []
        for fitting_indexes, test_indexes in tscv.split(fitting_data[0]):
            logger.warning(f"start cv: {step}")
            # collect
            gc.collect()
            model_copy = model
            # split training into a train and val test
            fitting_samples = len(fitting_indexes)
            last_train_index = int(fitting_samples * 0.8)
            train_indexes, val_indexes = fitting_indexes[:last_train_index], fitting_indexes[last_train_index:]

            # make the datasets
            train_set = (fitting_data[0][train_indexes], fitting_data[1][train_indexes])
            val_set = (fitting_data[0][val_indexes], fitting_data[1][val_indexes])
            test_set = (fitting_data[0][test_indexes], fitting_data[1][test_indexes])

            sets = (train_set, val_set, test_set)
            indexes = (train_index, movement_index, target_index)
            model_copy, loss_values = fit_and_predict(info_object=info_object,
                                                      logger=logger,
                                                      df=df,
                                                      model=model_copy,
                                                      n_out_steps=n_out_steps,
                                                      sets=sets,
                                                      indexes=indexes,
                                                      scaler=scaler,
                                                      step=step)
            step += 1
            loss_dicts.append(loss_values)

        # Mlflow log loss values
        losses = defaultdict(list)
        for d in loss_dicts:  # you can list as many input dicts as you want here
            for key, value in d.items():
                losses[key].append(value)
        avg_loss = {}
        for loss, vals in losses.items():
            avg_loss[f"avg_{loss}"] = mean(vals)

        mlflow.log_metrics(avg_loss)

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
