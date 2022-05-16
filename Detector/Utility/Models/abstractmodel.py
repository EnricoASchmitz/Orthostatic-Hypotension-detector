# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Abstract model implementation

# Imports
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
from optuna import Trial


class Model(ABC):
    def __init__(self):
        self.fig = False
        self.model = None
        self.parameters = None
        self.logger = logging.getLogger(__name__)
        self.m_eager = True

    @abstractmethod
    def get_copy(self):
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def fit(self, X_train_inputs: np.ndarray, y_train_outputs: np.ndarray, callbacks: list) -> int:
        """ Fit the model

        Args:
            train_set: Training set to use
            callbacks: callbacks to use

        Returns:
            number of iterations
        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def predict(self, data: np.ndarray) -> Union[np.ndarray, Optional[np.ndarray]]:
        """ Predict with model

        Args:
            data: test set to use
        Returns:
            Prediction, Optional[standard deviation]
        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def get_parameters(self):
        """ get parameters from model

        Returns:
            used parameters
        """

        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def set_parameters(self, arg: Union[dict, Trial]):
        """ Set parameters for model

        Args:
            arg: dictionary or trial to base parameters on

        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def _set_default_parameters(self):
        """ Set default parameters for model """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def _set_optuna_parameters(self, trail: Trial):
        """ Set parameters with optuna for model

        Args:
            trail: optuna trial

        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def save_model(self):
        """ save model """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def load_model(self, folder_name):
        """ load model """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")


def get_movement(past_mov, pred_y_xy):
    """ Get the future movement for the next step

    Args:
        past_mov: previous movement data to save the new movement to
        pred_y_xy: prediction for oxy and dxy used as a proximity for movement

    Returns:
        Movement data in the shape [predictions, timesteps, n_movement_features ]
    """
    # dxy is a good proxy for standing up
    oxy, dxy = pred_y_xy[0, ..., 0], pred_y_xy[0, ..., 1]
    oxy = np.expand_dims(oxy, axis=-1)
    dxy = np.expand_dims(dxy, axis=-1)
    oxy_reverse = -oxy + 1
    if oxy_reverse.ndim == 1:
        movement_data = np.concatenate([oxy_reverse, dxy])
        movement_data = np.expand_dims(movement_data, axis=0)
        past_mov = np.vstack([past_mov, movement_data])
    else:
        movement_data = np.mean(np.array([oxy_reverse, dxy]), axis=1).squeeze()
        past_mov = np.vstack([past_mov, movement_data])
    return past_mov
