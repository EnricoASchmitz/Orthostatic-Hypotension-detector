# Author: enrico schmitz (s1047521)
# master Thesis Data science
# Project: applications of Deep learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, cognition and Behaviour
# Script: Class to get loss values

# Imports
from statistics import mean

import numpy as np
from sklearn import metrics


class Loss:
    """ Object containing loss functions"""

    def __init__(self):
        self.loss_dict = {
            "mae": self.mean_absolute_error,
            "mape": self.mean_absolute_percentage_error,
            "mse": self.mean_squared_error,
            "rmse": self.root_mean_squared_error,
            "huber": self.huber,
            "log_cosh": self.log_cosh
        }

    @staticmethod
    def check_shape(true: np.ndarray, predicted: np.ndarray) -> bool:
        """ Check the shapes of the inputs

        Args:
            true: true values
            predicted: predicted values from models

        Raises:
            ValueError: if prediction or true values are missing
            AssertionError: if prediction and true values are of different shape

        Returns:
            If we need to apply loss recursively, True if more than 2 dims input.
        """
        if (predicted is None) or (true is None):
            raise ValueError("Missing prediction or true values")
        elif predicted.shape != true.shape:
            raise AssertionError("prediction and true values are different shape;"
                                 f" true shape = {true.shape}, pred shape = {predicted.shape}")
        elif predicted.ndim > 2:
            return True
        return False

    @staticmethod
    def _recursive_loss(true: np.ndarray, predicted: np.ndarray, function: callable) -> float:
        """ Apply a loss function over all multiple 2d frames

        Args:
            true: true values
            predicted: predicted values from models
            function: loss function to use

        Returns:
            average loss
        """
        loss_list = []
        # slice data
        for i in range(true.shape[1]):
            true_slice = true[:, i, :]
            predicted_slice = predicted[:, i, :]
            loss = function(true_slice, predicted_slice)
            loss_list.append(loss)
        return mean(loss_list)

    def get_loss_metric(self, loss_name: str) -> callable:
        """ Retrieve a single loss function using a name

        Args:
            loss_name: The name of the loss function to get.

        Returns:
            loss function
        """
        eval_func = self.loss_dict.get(loss_name, None)

        if eval_func is None:
            raise KeyError(f"Unknown evaluation metric function {loss_name}")

        return eval_func

    def get_loss_values(self, true: np.ndarray, predicted: np.ndarray, loss_list=None):
        """ Apply a loss function over all multiple 2d frames

        Args:
            true: true values
            predicted: predicted values from models
            loss_list: list with loss function to use, defaults to None uses all loss functions

        Returns:
            average loss
        """
        loss_values = {}
        # subset dict with loss_list
        if loss_list is not None:
            loss_dict = {key: self.loss_dict[key] for key in loss_list}
        else:
            loss_dict = self.loss_dict
        for name, loss_function in loss_dict.items():
            loss_values[name] = loss_function(true=true, predicted=predicted)
        return loss_values

    def mean_absolute_error(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Mean absolute error

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.mean_absolute_error)
        return metrics.mean_absolute_error(true, predicted)

    def mean_absolute_percentage_error(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Mean absolute percentage error

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.mean_absolute_percentage_error)
        return metrics.mean_absolute_percentage_error(true, predicted)

    def mean_squared_error(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Mean squared error

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.mean_squared_error)
        return metrics.mean_squared_error(true, predicted, squared=True)

    def root_mean_squared_error(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Root mean squared error

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.root_mean_squared_error)
        return metrics.mean_squared_error(true, predicted, squared=False)

    def huber(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Huber loss

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.huber)

        loss = self._huber_loss(true, predicted)
        return loss

    @staticmethod
    def _huber_loss(true: np.ndarray, predicted: np.ndarray) -> float:
        threshold = 1
        e = true - predicted
        is_low = np.abs(e) <= threshold
        loss_small_error = np.square(e) / 2
        loss_big_error = threshold * (np.abs(e) - (0.5 * threshold))
        return float(np.sum(np.where(is_low, loss_small_error, loss_big_error)))

    def log_cosh(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """ Log-Cosh loss

        Args:
            true: true values
            predicted: predicted values from models

        Returns:
            Loss value
        """
        multi_dim = self.check_shape(true, predicted)
        if multi_dim:
            return self._recursive_loss(true, predicted, self.log_cosh)

        loss = self._log_cosh(true, predicted)
        return loss

    @staticmethod
    def _log_cosh(true: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.sum(np.log(np.cosh(predicted - true))))
