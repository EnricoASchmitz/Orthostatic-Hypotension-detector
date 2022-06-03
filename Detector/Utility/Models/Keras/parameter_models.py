# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras parameter model implementations

# Imports

import optuna
from tensorflow.keras.layers import Reshape, Conv1D

from Detector.Utility.Models.Keras.kerasmodel import Base
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


class Dense(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create LSTM model

        Args:
            data_object: information retrieved from the data
            input_shape: Shape of input
            output_shape: Shape of output
            gpu: bool, if GPU is available
            plot_layers: bool, plot the models
            parameters: Parameters to use for model creation
        """
        super().__init__(data_object=data_object,
                         input_shape=input_shape,
                         output_shape=output_shape,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def reshape_layer(self, input_layer, **kwargs):
        re = Reshape((-1,))(input_layer)
        bp_out = self._output_layers_parameters(re, **kwargs)
        return bp_out

    def _get_model(self):
        return self.reshape_layer

    def _set_default_parameters(self):
        model_parameters = {
            "n_dense_layers": 0,
            "dropout": 0.0,
            "activation_out": "linear",
            "batch_norm": True
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get Dense parameters
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        if n_dense_layers > 0:
            dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        else:
            dropout = 0
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "batch_norm": batch_norm,
        }
        self.parameters.update(model_parameters)


class Cnn(Base):
    """ Basic CNN model """

    # todo fix Call to CreateProcess failed. Error code: 2
    def __init__(self, data_object: DataObject, input_shape: tuple,
                 output_shape: tuple, gpu: bool, plot_layers=False,
                 parameters=None):
        """ Create CNN model

        Args:
            data_object: information retrieved from the data
            input_shape: Shape of input
            output_shape: Shape of output
            gpu: bool, if GPU is available
            plot_layers: bool, plot the models
            parameters: Parameters to use for model creation
        """
        super().__init__(data_object=data_object,
                         input_shape=input_shape,
                         output_shape=output_shape,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def cnn_layer(self, input_layer, units_layer, kernel_size, **kwargs):
        cnn_layer = Conv1D(filters=int(units_layer), kernel_size=int(kernel_size), padding="same")(input_layer)
        re = Reshape((-1,))(cnn_layer)
        bp_out = self._output_layers_parameters(re, **kwargs)
        return bp_out

    def _get_model(self):
        return self.cnn_layer

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "kernel_size": 5,
            "n_dense_layers": 0,
            "dropout": 0,
            "activation_out": "linear",
            "batch_norm": True
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get CNN parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        kernel_size = trial.suggest_int("kernel_size", 2, 6, step=1)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
            "kernel_size": kernel_size,
            "batch_norm": batch_norm,
        }
        self.parameters.update(model_parameters)
