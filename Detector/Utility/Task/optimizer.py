# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Optuna optimizing

# Imports
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from keras.backend import clear_session
from optuna import Study
from optuna.integration import TFKerasPruningCallback, XGBoostPruningCallback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from Detector.Utility.Data_preprocessing.Outlier_detection.Outlier_detection import outlier_detection
from Detector.Utility.Metrics.Losses import Loss
from Detector.Utility.Models.Decision_trees.XGBoost import XGB
from Detector.Utility.Models.Keras.kerasmodel import KerasModel
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Task.model_functions import check_gpu, fitting, predicting
from Detector.Utility.Task.setup_data import prepare_data, inverse_scale
from Detector.enums import Parameters


class Optimizer:
    """ Optimize a model with optuna """

    def __init__(self, df: pd.DataFrame, info_object: InfoObject, data_object: DataObject, scaler: MinMaxScaler,
                 n_in_steps: int, n_out_steps: int, n_features: int):
        self.scaler = scaler
        self.n_features = n_features
        self.n_out_steps = n_out_steps
        self.n_in_steps = n_in_steps
        self.df = df
        self.info_object = info_object
        self.data_object = data_object
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def load_study(self, storage) -> Optional[Study]:
        """ Load the previous optuna study if it exists

        Args:
            storage: the file location

        Returns:
            if the storage file exists and contains the study, return loaded study
        """
        file = os.path.basename(storage)
        file_path = Path(file.removeprefix('sqlite:///'))
        # check if the file exists
        if file_path.exists():
            try:
                loaded_study = optuna.load_study(study_name=self.info_object.model, storage=storage)
                self.logger.info("loading study")
                return loaded_study
            # can happen that it doesn't contain a study from our current model, so start clean
            except KeyError:
                return
        else:
            return

    def optimize_parameters(self, storage: str) -> Study:
        """ Perform a study with optuna

        Returns:
            optuna study

        """
        storage = f"sqlite:///{storage}"
        # check for old study
        study = self.load_study(storage)
        # Optuna
        if study is None:
            study = optuna.create_study(study_name=self.info_object.model, direction="minimize", storage=storage)
            self.logger.info("creating study")

        study.optimize(self._objective,
                       n_trials=Parameters.n_trials.value,
                       gc_after_trial=True,
                       callbacks=[lambda study, trial: clear_session()]
                       )
        return study

    def _objective(self, trial: optuna.trial.Trial) -> Tuple[float, float]:
        """ Optuna objective function to perform for each trial

        Args:
            trial: Optuna trial

        Returns:
            mae loss score
        """
        use_gpu = check_gpu()

        try:
            # make a model
            n_in_features = self.n_features - len(self.data_object.movement_features) - len(
                self.data_object.target_col)
            # movement
            n_mov_features = len(self.data_object.movement_features)
            model = ModelCreator.create_model(self.info_object.model,
                                              data_object=self.data_object,
                                              n_in_steps=self.n_in_steps,
                                              n_out_steps=self.n_out_steps,
                                              n_in_features=n_in_features,
                                              n_mov_features=n_mov_features,
                                              gpu=use_gpu,
                                              parameters=trial
                                              )
            # get data
            # filter data with a OD algorithm
            # outlier detection
            algorithms = [None, "knn", "iqr", "if", "lof"]
            od_alg = trial.suggest_categorical("Outlier detection", algorithms)
            if od_alg is not None:
                self.df = outlier_detection(algo_name=od_alg, df=self.df, columns=list(self.df.columns))
                self.df.drop(columns="outliers", inplace=True)

            train_index, target_index, input_data = prepare_data(model,
                                                                 self.df,
                                                                 self.data_object,
                                                                 val_set=True,
                                                                 test_set=True
                                                                 )
            train_set, test_set, val_set = input_data
            fit_X = np.vstack([train_set[0], val_set[0]])
            fit_y = np.vstack([train_set[1], val_set[1]])
            fitting_set = [fit_X, fit_y]
            if isinstance(model, KerasModel):
                callbacks = [TFKerasPruningCallback(trial, "val_loss")]
            elif isinstance(model, XGB):
                callbacks = [XGBoostPruningCallback(trial, observation_key=f"validation_0-mae")]
            else:
                callbacks = []

            # fit model
            movement_index = {}
            for movement_col in self.data_object.movement_features:
                movement_index[movement_col] = train_index[movement_col]
            n_iterations, training_time = fitting(model=model, logger=self.logger, train_set=train_set, val_set=val_set,
                                                  movement_index=movement_index, callbacks=callbacks)

            # predict with model
            prediction, std, predict_time = predicting(model=model, logger=self.logger, test_set=test_set,
                                                       n_out_steps=self.n_out_steps, movement_index=movement_index)

            # inverse scaling
            train, test, prediction = inverse_scale(scaler=self.scaler, train=fitting_set,
                                                    test=test_set, prediction=prediction)

            del model
            # return mae
            mae = Loss().get_loss_metric("mae")
            mae = round(mae(test, prediction), 4)
        except ValueError as e:
            self.logger.warning(e)
            mae = 1e+10
        except ResourceExhaustedError as ie:
            self.logger.warning("OOM")
            self.logger.warning(f"Parameters= {model.get_parameters()}")
            mae = 1e+10
        return mae


@contextmanager
def suppress_stdout():
    """ Mute output """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
