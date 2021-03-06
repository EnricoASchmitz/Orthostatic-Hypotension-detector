# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras parameter model implementations

# Imports
from abc import abstractmethod
from typing import Optional

import numpy as np
import optuna
from keras.layers import Flatten, AveragePooling1D
from optuna import Trial
from tensorflow import get_logger
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.utils.vis_utils import plot_model

from Detector.Utility.Models.Keras.kerasmodel import KerasModel
from Detector.enums import Parameters


class Base(KerasModel):
    def __init__(self, input_shape, output_shape, parameters: Optional[dict],
                 plot_layers: bool,
                 gpu):
        get_logger().setLevel("ERROR")
        super().__init__()
        self.logger.debug(f"GPU: {gpu}")

        self.input_shape = input_shape
        self.output_shape = output_shape

        # fill parameters
        self.set_parameters(parameters)

        self.model = self._model(**self.parameters)
        if plot_layers:
            plot_model(self.model, show_shapes=True, to_file="model.png")

    def _model(self, optimizer: str, loss: str,
               **kwargs):
        """ Make model

       Args:
            optimizer: optimizer
            loss: loss
       """
        # Creating the layers
        inputs = Input(shape=self.input_shape,
                       name="BP_in")
        last_layer = BatchNormalization()(inputs)
        model_loss = None

        model = self.compile_model(self._get_model(), inputs, last_layer, optimizer, loss, model_loss, **kwargs)
        return model

    def _output_layers_parameters(self, prev_layer, n_dense_layers, dropout, activation_out, batch_norm,
                                  **kwargs) -> KerasTensor:
        """ Create output layers

        Args:
             prev_layer: previous layer
             n_dense_layers: number of Dense layers to use
             dropout: dropout value
             activation: activation to use for final layer
        """
        n_dense_layers = int(n_dense_layers)
        dropout_value = float(dropout)

        out_units = self.output_shape[-1]
        input_shape = prev_layer.shape[-1]

        layer_units = np.flip(np.linspace(out_units, input_shape, num=n_dense_layers))
        for i in range(n_dense_layers):
            dense_layer = Dense(layer_units[i], activation="relu", name=f"Dense_{i}")(prev_layer)
            if batch_norm:
                layer = BatchNormalization(name=f"BN_{i}")(dense_layer)
            else:
                layer = dense_layer
            prev_layer = Dropout(dropout_value, name=f"dropout_out_{i}")(layer)
        out_layer = Dense(units=out_units, name="BP_out", activation=activation_out)(prev_layer)
        return out_layer

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError("Must override method")

    def _set_optuna_parameters(self, trial: Trial):
        pass

    def _set_default_parameters(self):
        pass


class MLP(Base):
    def __init__(self, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create MLP model

        Args:
            input_shape: Shape of input
            output_shape: Shape of output
            gpu: bool, if GPU is available
            plot_layers: bool, plot the models
            parameters: Parameters to use for model creation
        """
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def reshape_layer(self, input_layer, **kwargs):
        re = Flatten()(input_layer)
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
            batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        else:
            dropout = 0
            batch_norm = False
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
    def __init__(self, input_shape: tuple,
                 output_shape: tuple, gpu: bool, plot_layers=False,
                 parameters=None):
        """ Create CNN model

        Args:
            input_shape: Shape of input
            output_shape: Shape of output
            gpu: bool, if GPU is available
            plot_layers: bool, plot the models
            parameters: Parameters to use for model creation
        """
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def cnn_layer(self, input_layer, filters, kernel_size, pooling, pool_size, strides, **kwargs):
        last_layer = Conv1D(filters=int(filters), kernel_size=int(kernel_size), strides=int(strides))(input_layer)
        if pooling:
            last_layer = AveragePooling1D(int(pool_size))(last_layer)
        re = Flatten()(last_layer)
        bp_out = self._output_layers_parameters(re, **kwargs)
        return bp_out

    def _get_model(self):
        return self.cnn_layer

    def _set_default_parameters(self):
        model_parameters = {
            "filters": 64,
            "kernel_size": 5,
            "pooling": True,
            "pool_size": 2,
            "strides": 1,
            "n_dense_layers": 0,
            "dropout": 0,
            "activation_out": "linear",
            "batch_norm": True
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get CNN parameters
        filters = trial.suggest_int("filters", 16, Parameters.default_units.value * 2)
        kernel_size = trial.suggest_int("kernel_size", 2, 10)
        pooling = trial.suggest_categorical("pooling", [True, False])
        if pooling:
            pool_size = trial.suggest_int("pool_size", 2, 6, step=2)
        else:
            pool_size = 0
        strides = trial.suggest_int("strides", 1, 10)

        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "filters": filters,
            "strides": strides,
            "pooling": pooling,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "batch_norm": batch_norm,
        }

        self.parameters.update(model_parameters)
