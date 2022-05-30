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
from sklearn.model_selection import train_test_split

from Detector.Utility.Data_preprocessing.Transformation import scale2d, scale3d, reverse_scale2d, reverse_scale3d
from Detector.Utility.Data_preprocessing.extract_info import parameters_to_curve, make_curves
from Detector.Utility.Metrics.Losses import Loss
from Detector.Utility.Models.Decision_trees.XGBoost import XGB
from Detector.Utility.Models.Keras.kerasmodel import KerasModel
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.Plotting import plotting
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Task.model_functions import check_gpu
from Detector.enums import Parameters


class Optimizer:
    """ Optimize a model with optuna """

    def __init__(self, x, output, info_object: InfoObject, data_object: DataObject):

        self.input = x
        self.output = output
        self.info_object = info_object
        self.data_object = data_object
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def load_study(self, storage) -> Optional[Study]:
        """ Load the previous optuna study if it exists

        Args:
            storage: the file location

        Returns:
            if the storage file exists and contains the study, return loaded study
        """
        file = os.path.basename(storage)
        file_path = Path(file.removeprefix("sqlite:///"))
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
            loss score
        """
        use_gpu = check_gpu()

        X_unscaled = self.input
        X, x_scalers = scale3d(X_unscaled.copy(), self.data_object)

        if self.output.ndim == 2:
            output_unscaled = np.array(self.output)
            out_scale_function = scale2d
            out_reverse_scale_function = reverse_scale2d
        else:
            output_unscaled = np.array(self.output)
            out_scale_function = scale3d
            out_reverse_scale_function = reverse_scale3d

        output, out_scaler = out_scale_function(output_unscaled.copy(), self.data_object)

        train, test = train_test_split(range(len(X)))
        try:
            model = ModelCreator.create_model(self.info_object.model, data_object=self.data_object,
                                              input_shape=X.shape[1:],
                                              output_shape=output.shape[1:],
                                              gpu=use_gpu,
                                              parameters=trial)

            if isinstance(model, KerasModel):
                callbacks = [TFKerasPruningCallback(trial, "val_loss")]
            elif isinstance(model, XGB):
                callbacks = [XGBoostPruningCallback(trial, observation_key=f"validation_0-{Parameters.loss.value}")]
            else:
                callbacks = []

            model.fit(X[train], output[train], callbacks)
            prediction, std = model.predict(X[test])

            # Scale back the prediction
            prediction = out_reverse_scale_function(prediction, out_scaler)

            del model

            # Get loss function
            loss = Loss().get_loss_metric(Parameters.loss.value)

            # Compare parameter to full curve:
            if self.info_object.parameter_model:
                prediction = pd.DataFrame(prediction, columns=self.output.columns).copy()
                true_curves, pred_curves = make_curves(prediction, self.output, self.data_object.reconstruct_params, self.data_object.recovery_times, test)
                loss_value = round(loss(true_curves, pred_curves), 4)

                # add the parameters loss not used for reconstruction
                extra_params_pred = prediction.drop(self.data_object.reconstruct_params, axis=1)
                extra_params_true = self.output.copy().drop(self.data_object.reconstruct_params, axis=1).iloc[test]
                loss_value += round(loss(extra_params_true, extra_params_pred), 4)
            else:
                loss_value = round(loss(output_unscaled[test], prediction), 4)
            if loss_value is np.nan:
                loss_value = 1e+10
        except KeyError as e:
            self.logger.error(e)
            loss_value = 1e+10
        return loss_value
