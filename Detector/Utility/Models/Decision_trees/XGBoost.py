# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: XGBoost regressor implementation

# Imports
import logging
import os.path
import pickle
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from optuna import Trial
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import has_fit_parameter, _check_fit_params
from tqdm import tqdm
from xgboost import XGBRegressor

from Detector.Utility.Models.abstractmodel import Model, get_movement
from Detector.Utility.PydanticObject import DataObject
from Detector.Utility.Task.setup_data import split_series
from Detector.enums import Parameters


class XGB(Model):
    """ XGBoost model """

    def __init__(self, data_object: DataObject, gpu, n_out_steps, n_in_features, n_mov_features, parameters=None,
                 **kwargs):
        """ Create XGBoost model """
        super().__init__()
        self.n_in_steps = None
        if gpu:
            logger = logging.getLogger()
            logger.debug("using GPU")
            tree_method = 'gpu_hist'
        else:
            tree_method = 'hist'
        self.data_object = data_object
        self.multi_step_in = False
        self.multi_step_out = False
        self.n_in_steps = 1
        self.n_out_steps = n_out_steps
        self.n_in_features = n_in_features
        self.n_mov_features = n_mov_features
        self.n_features = n_in_features + n_mov_features
        # fill parameters
        self.set_parameters(parameters)

        xgb = XGBRegressor(tree_method=tree_method, eval_metric='mae', verbosity=1, **self.parameters)

        model = MyMultiOutputRegressor(xgb)

        self.model = model

    def get_data(self, data, train_index, target_index, val_set, test_set):
        timeserie_X, timeserie_y = split_series(data, self.n_in_steps, 1, train_index, target_index)
        if val_set or test_set:
            data_X = self._split_data(timeserie_X, val_set, test_set)
            data_y = self._split_data(timeserie_y, val_set, test_set)
            data_output = []
            for set_i in range(len(data_X)):
                data_output.append((data_X[set_i], data_y[set_i]))

            return data_output
        else:
            return timeserie_X, timeserie_y

    def _split_data(self, timeserie, val_set: bool, test_set: bool):
        datasets = []
        train_set = None
        if test_set:
            train_set, test_set = train_test_split(timeserie, test_size=0.2, random_state=1, shuffle=False)
            test_start = train_set[-self.n_in_steps:]
            test_with_start = np.concatenate((test_start, test_set), axis=0)
            datasets.append(test_with_start.squeeze())
        if val_set:
            if train_set is None:
                train_set = timeserie
            train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=1, shuffle=False)
            datasets.append(val_set.squeeze())
        datasets.append(train_set.squeeze())
        # the data set will start with the test set and then all other sets
        datasets.reverse()
        return datasets

    def fit(self, train_set: Tuple[np.ndarray, np.ndarray], val_set: Tuple[np.ndarray, np.ndarray],
            callbacks: list, **kwargs) -> int:
        """ Fit XGBoost model

        :param train_set: timeseries training data
        :param val_set: timeseries validation data
        :param callbacks: callbacks to use
        :return: number of used estimators
        """
        self.model.fit(train_set[0], train_set[1], eval_set=[val_set], eval_metric="mae", callbacks=callbacks,
                       verbose=0)
        return self.parameters["n_estimators"]

    def predict(self, data, **kwargs):
        if isinstance(data, tuple):
            data = data[0]

        data = data.squeeze()
        if data.ndim == 1:
            data = data.reshape((1, len(data)))
        assert data.ndim == 2, "dimensions incorrect"
        prediction = self.model.predict(data)

        return prediction, None

    def predict_future(self, data, num_prediction):

        prediction_list = data[-1]
        out_list = []
        Output_full = []
        past_mov = prediction_list[self.n_in_features:]
        past_mov = past_mov.reshape(1, self.n_mov_features)
        for _ in tqdm(range(num_prediction)):
            if prediction_list.ndim == 2:
                x = prediction_list[-1:]
            else:
                x = prediction_list.copy()
            x = x.reshape((1, self.n_features))
            out, _ = self.predict(x)
            # save output
            output = list(out[:, :len(self.data_object.target_col)])
            out_features = list(out[:, len(self.data_object.target_col):])
            Output_full.append(output)
            out_list.append(out_features)
            # remove features that are not needed for the input
            remove_index = list(range(0, len(self.data_object.target_col)))
            input_next_run = np.delete(out, remove_index, axis=1)
            # add movement
            past_mov = get_movement(past_mov, input_next_run)
            next_mov = past_mov[-self.n_in_steps:]

            input_next_run = np.hstack([input_next_run, next_mov])
            input_next_run = input_next_run.reshape((1, self.n_features))
            prediction_list = np.vstack([prediction_list, input_next_run])
        # add output to the prediction
        Output_full = np.array(Output_full)
        out_list = np.array(out_list)
        prediction_list = np.dstack((Output_full, out_list)).squeeze()
        return prediction_list, None

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
            "objective": 'reg:squarederror'
        }

    def _set_optuna_parameters(self, trial):
        param = {
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 4, 31, step=4)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        self.parameters.update(param)

    def save_model(self):
        pickle.dump(self.model, open("model.pkl", "wb"))

    def load_model(self, folder_name):
        file_name = os.path.join(folder_name, "model.pkl")
        self.model = pickle.load(open(file_name, "rb"))


class MyMultiOutputRegressor(MultiOutputRegressor):
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, **fit_params) -> None:
        """ Fit the model to data.
        Fit a separate model for each output variable.
        Allows us to fit a validation set.

        :param x: {array-like, sparse matrix} of shape (n_samples, n_features)
        Data.
        :param y: {array-like, sparse matrix} of shape (n_samples, n_outputs)
        Multi-output targets. An indicator matrix turns on multilabel
        estimation.
        :param sample_weight: array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.
        Only supported if the underlying regressor supports sample
        weights.
        :param fit_params: dict of string -> object
        Parameters passed to the ``estimator.fit`` method of each step.
        :return: None
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
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)
        [(X_test, Y_test)] = fit_params_validated.pop('eval_set')
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
