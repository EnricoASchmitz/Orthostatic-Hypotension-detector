# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: XGBoost regressor implementation

# Imports
import logging
import os.path
import pickle

import numpy as np
from joblib import Parallel, delayed
from optuna import Trial
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import has_fit_parameter, _check_fit_params
from xgboost import XGBRegressor

from Detector.Utility.Models.abstractmodel import Model
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


class XGB(Model):
    """ XGBoost model """

    def __init__(self, data_object: DataObject, gpu, parameters=None,
                 **kwargs):
        """ Create XGBoost model """
        super().__init__()
        self.n_in_steps = None
        self.gpu = gpu
        if gpu:
            logger = logging.getLogger()
            logger.debug("using GPU")
            self.parameters.update({"tree_method": "gpu_hist"})
        self.data_object = data_object

        # fill parameters
        self.set_parameters(parameters)

        xgb = XGBRegressor(eval_metric=Parameters.loss.value, verbosity=1, **self.parameters)

        model = MyMultiOutputRegressor(xgb)

        self.model = model

    def fit(self, x_train_inputs, y_train_outputs, callbacks, **kwargs):
        index_train, index_val = train_test_split(range(len(x_train_inputs)), test_size=0.33, random_state=42)

        # reshape X to 2d by adding timesteps as a feature
        x_train_inputs = self._add_timestep_as_feature(x_train_inputs)

        X_train, X_val = x_train_inputs[index_train], x_train_inputs[index_val]
        y_train, y_val = y_train_outputs[index_train], y_train_outputs[index_val]

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=Parameters.loss.value,
                       callbacks=callbacks,
                       verbose=0)
        return self.parameters["n_estimators"]

    @staticmethod
    def _add_timestep_as_feature(x: np.ndarray) -> np.ndarray:
        """ Make 3D dataframe to 2D

        Args:
            x: 3D dataframe, will add all time steps as features

        """
        return x.reshape(x.shape[0], -1)

    def predict(self, data, **kwargs):
        data = self._add_timestep_as_feature(data)
        assert data.ndim == 2, "dimensions incorrect"
        prediction = self.model.predict(data)

        return prediction, None

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, arg=None):
        if arg is None:
            self.logger.info("setting default variables")
            self._set_default_parameters()
        elif isinstance(arg, dict):
            self.logger.info("updating default variables with parameters")
            self._set_default_parameters()
            parameters = self.parameters
            arg.update(parameters)
            try:
                arg.pop("Outlier detection")
            except KeyError:
                pass
            self.parameters = arg
        else:
            assert isinstance(arg, Trial), "wrong format of arguments"
            self.logger.info("Setting with optuna")
            self._set_default_parameters()
            self._set_optuna_parameters(arg)

    def _set_default_parameters(self, parameters=None):
        self.parameters = {
            "n_estimators": (Parameters.iterations.value * 10),
            "objective": "reg:squarederror"
        }

    def _set_optuna_parameters(self, trial):
        param = {
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1, step=0.1),
            "max_bin": trial.suggest_int("max_bin", 64, 512, step=32),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }
        if self.gpu:
            param["sampling_method"] = trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"]),

            if param["sampling_method"] == "gradient_based":
                param["subsample"] = trial.suggest_float("subsample", 0.5, 1, step=0.1)
        elif not self.gpu or param["sampling_method"] == "uniform":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, step=0.1)
        self.parameters.update(param)

    def save_model(self):
        pickle.dump(self.model, open("model.pkl", "wb"))

    def load_model(self, folder_name):
        file_name = os.path.join(folder_name, "model.pkl")
        self.model = pickle.load(open(file_name, "rb"))


class MyMultiOutputRegressor(MultiOutputRegressor):
    def __init__(self, estimator):
        super().__init__(estimator)
        self.estimators_ = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, **fit_params) -> None:
        """ Fit the model to data.
        Fit a separate model for each output variable.
        Allows us to fit a validation set.

        Args:
            x: {array-like, sparse matrix} of shape (n_samples, n_features) Data.

            y: {array-like, sparse matrix} of shape (n_samples, n_outputs) Multi-output targets.

            sample_weight: array-like of shape (n_samples,), default=None

            fit_params: dict of string -> object; Parameters passed to the ``estimator.fit`` method of each step.

        Returns:
            None
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             " a fit method")

        X, y = self._validate_data(x, y,
                                   force_all_finite=False,
                                   multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, "sample_weight")):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)
        [(X_test, Y_test)] = fit_params_validated.pop("eval_set")
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator,
                X,
                y[:, i],
                sample_weight,
                **fit_params_validated,
                eval_set=[(X_test, Y_test[:, i])]
            )
            for i in range(y.shape[1]))
