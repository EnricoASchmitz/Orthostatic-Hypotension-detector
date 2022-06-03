# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Optuna optimizing

# Imports
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import optuna
from keras.backend import clear_session
from optuna import Study
from optuna.integration import TFKerasPruningCallback, XGBoostPruningCallback

from Detector.Utility.Data_preprocessing.Transformation import scale2d, scale3d, reverse_scale2d
from Detector.Utility.Models.XGBoost import XGB
from Detector.Utility.Models.Keras.kerasmodel import KerasModel
from Detector.Utility.Models.Model_creator import ModelCreator
from Detector.Utility.PydanticObject import InfoObject, DataObject
from Detector.Utility.Task.model_functions import check_gpu, fit_and_predict, filter_out_test_subjects
from Detector.enums import Parameters


class Optimizer:
    """ Optimize a model with optuna """

    def __init__(self, x, output, info_dataset, info_object: InfoObject, data_object: DataObject):

        self.input = x
        self.output = output
        self.info_dataset = info_dataset
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

        output_unscaled = np.array(self.output)
        output, out_scaler = scale2d(output_unscaled.copy(), self.data_object)

        indexes = filter_out_test_subjects(self.info_dataset)
        try:
            model = ModelCreator.create_model(self.info_object.model,
                                              input_shape=X.shape[1:],
                                              output_shape=output.shape[1:],
                                              gpu=use_gpu, plot_layers=True,
                                              parameters=trial)
            if isinstance(model, KerasModel):
                callbacks = [TFKerasPruningCallback(trial, "val_loss")]
            elif isinstance(model, XGB):
                loss = Parameters.loss.value
                if loss == "mse":
                    loss = "rmse"
                callbacks = [XGBoostPruningCallback(trial, observation_key=f"validation_0-{loss}")]
            else:
                callbacks = []
            model, loss_value = fit_and_predict(info_object=self.info_object,
                                                logger=self.logger,
                                                input_values=X,
                                                output_values=output,
                                                model=model,
                                                step=0,
                                                indexes=indexes,
                                                scaler=out_scaler,
                                                rescale_function=reverse_scale2d,
                                                callbacks=callbacks,
                                                loss_function=Parameters.loss.value)
            del model

        except ValueError as e:
            self.logger.warning(e)
            loss_value = 1e+10
        except AssertionError as e:
            self.logger.warning(e)
            loss_value = 1e+10
        if loss_value is np.nan:
            loss_value = 1e+10

        return loss_value
