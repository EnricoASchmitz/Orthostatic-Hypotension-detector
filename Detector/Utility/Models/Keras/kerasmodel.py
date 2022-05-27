# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras general model implementation

# Imports
import os
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from optuna import Trial
from tensorflow import keras, get_logger
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.utils.vis_utils import plot_model

from Detector.Utility.Models.abstractmodel import Model
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


class KerasModel(Model):
    def __init__(self):
        super().__init__()

        self.get_intermediate = None
        self.model = None
        self.data_object = None
        self.input_shape = None
        self.output_shape = None
        self.optimizer = None
        self.model_loss = None
        self.fig = True

    def compile_model(self, model_function: callable, inputs: KerasTensor, prev_layer: KerasTensor,
                      optimizer: str, loss: str, model_loss: callable, **kwargs):
        """ Compile the model

        Args:
            model_function: function which add layers
            inputs: input layer
            prev_layer: last layer
            optimizer: optimizer
            loss: loss
            model_loss: model_loss function

        """
        bp_out = model_function(prev_layer, **kwargs)
        # model
        model = keras.Model(inputs=inputs, outputs=bp_out,
                            name='BP_model')
        if self.m_eager:
            run_eager = True
        else:
            run_eager = False
        if model_loss is None:
            model_loss = loss

        self.optimizer = optimizer
        self.model_loss = model_loss

        model.compile(optimizer=optimizer, loss=model_loss, metrics=['mae'], run_eagerly=run_eager)

        return model

    def fit(self, x_train_inputs: np.ndarray, y_train_outputs: np.ndarray, callbacks: list) -> int:

        early_stopper = EarlyStopping(patience=10, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(factor=0.33, patience=6, min_lr=1e-6, verbose=0)
        nan_terminate = TerminateOnNaN()

        if callbacks is None:
            callbacks = [early_stopper, lr_reducer, nan_terminate]
        else:
            callbacks.extend([early_stopper, lr_reducer, nan_terminate])

        split = float(Parameters.validation_split.value)

        history = self.model.fit(x=x_train_inputs, y=y_train_outputs,
                                 validation_split=split,
                                 epochs=self.parameters["epochs"],
                                 batch_size=self.parameters["batch_size"],
                                 shuffle=True,
                                 callbacks=callbacks,
                                 verbose=0)
        # get intermediate values from model if needed
        self.get_intermediate = self.get_intermediate_values(self.model)

        return len(history.history["loss"])

    def _make_prediction(self, inputs):
        if self.get_intermediate:
            model_pred, std = self.use_intermediate_values(inputs, self.get_intermediate)
        else:
            model_pred = self.model.predict(x=inputs)
            std = None
        return model_pred, std

    def predict(self, data):
        # use intermediate values for making the prediction
        prediction, std = self._make_prediction(inputs=data)
        return prediction, std

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, arg=None):
        self.parameters = self.get_keras_parameters(arg)
        if arg is None:
            self.logger.info("setting default variables")
            self._set_default_parameters()
        elif isinstance(arg, dict):
            self.logger.info("updating default variables with parameters")
            self._set_default_parameters()
            parameters = self.parameters
            parameters.update(arg)
            try:
                parameters.pop("Outlier detection")
            except KeyError:
                pass
            self.parameters = parameters
        else:
            self.logger.info("Setting with optuna")
            self._set_optuna_parameters(arg)

    def _set_optuna_parameters(self, trial: Trial):
        raise NotImplementedError("This is model specific, needs to be called from model class")

    def _set_default_parameters(self):
        raise NotImplementedError("This is model specific, needs to be called from model class")

    @staticmethod
    def get_keras_parameters(args: Optional[Union[dict, Trial]] = None) -> dict:
        """ Get keras specific parameters

        Args:
            args: optional dictionary with parameters or optuna trial

        Returns:
            dictionary with keras parameters
        """
        epochs = Parameters.iterations.value
        batch_size = Parameters.batch_size.value
        if isinstance(args, Trial):
            opti = args.suggest_categorical("optimizer", ["SGD", "RMSprop", "Adam"])
            if opti == "SGD":
                opti = SGD
            elif opti == "RMSprop":
                opti = RMSprop
            else:
                opti = Adam

            loss = "mae"
            args.set_user_attr("loss", loss)

            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": opti(),
                "loss": loss
            }
        elif isinstance(args, dict):
            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": args["optimizer"],
                "loss": args["loss"]
            }
        else:

            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "adam",
                "loss": "mse"
            }
        return keras_parameters

    def save_model(self):
        self.model.save_weights("model.h5")

    def load_model(self, folder_name):
        # contains bug
        file_name = os.path.join(folder_name, "model.h5")
        self.model.load_weights(file_name)
        # get intermediate values from model if needed
        self.get_intermediate = self.get_intermediate_values(self.model)

    def get_intermediate_values(self, model):
        pass

    def use_intermediate_values(self, inputs, get_intermediate) -> Union[np.ndarray, np.ndarray]:
        pass


class Base(KerasModel):
    def __init__(self, data_object: DataObject, input_shape, output_shape, parameters: Optional[dict],
                 plot_layers: bool,
                 gpu):
        get_logger().setLevel('ERROR')
        super().__init__()
        self.logger.debug(f"GPU: {gpu}")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_object = data_object

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
                       name='BP_in')
        last_layer = BatchNormalization()(inputs)
        model_loss = None

        model = self.compile_model(self._get_model(), inputs, last_layer, optimizer, loss, model_loss, **kwargs)
        return model

    def _output_layers_parameters(self, prev_layer, n_dense_layers, dropout, activation_out, **kwargs) -> KerasTensor:
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
            dense_layer = Dense(layer_units[i], activation="relu", name=f'Dense_{i}')(prev_layer)
            BN = BatchNormalization(name=f"BN_{i}")(dense_layer)
            prev_layer = Dropout(dropout_value, name=f'dropout_out_{i}')(BN)
        out_layer = Dense(units=out_units, name='BP_out', activation=activation_out)(prev_layer)
        return out_layer

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError("Must override method")

    def _set_optuna_parameters(self, trial: Trial):
        pass

    def _set_default_parameters(self):
        pass
