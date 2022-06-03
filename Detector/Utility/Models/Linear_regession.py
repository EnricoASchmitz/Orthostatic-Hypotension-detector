# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: XGBoost regressor implementation

# Imports
import os.path
import pickle

import numpy as np
from optuna import Trial
from sklearn.linear_model import LinearRegression

from Detector.Utility.Models.abstractmodel import Model


class LinearRegressor(Model):
    """ Linear regression model """

    def __init__(self, **kwargs):
        """ Create linear regression model """
        super().__init__()
        # fill parameters
        self.model = LinearRegression()

    def fit(self, x_train_inputs, y_train_outputs, callbacks, **kwargs):
        # reshape X to 2d by adding timesteps as a feature
        x_train_inputs = self._add_timestep_as_feature(x_train_inputs)
        self.model.fit(x_train_inputs, y_train_outputs)
        return 0

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

        return prediction

    def get_parameters(self):
        return {}

    def set_parameters(self, arg=None):
        """ For compatibility """
        pass

    def _set_default_parameters(self):
        """ For compatibility """
        pass

    def _set_optuna_parameters(self, trail: Trial):
        """ For compatibility """
        pass

    def save_model(self):
        pickle.dump(self.model, open("model.pkl", "wb"))

    def load_model(self, folder_name):
        file_name = os.path.join(folder_name, "model.pkl")
        self.model = pickle.load(open(file_name, "rb"))
