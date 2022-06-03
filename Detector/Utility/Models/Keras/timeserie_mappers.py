# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script:

# Imports

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.python.keras.utils.vis_utils import plot_model

from Detector.Utility.Models.Keras.kerasmodel import KerasModel


class TimeserieMLP(KerasModel):
    def __init__(self, input_shape, output_shape, parameters, plot_layers=False, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.set_parameters(parameters)
        self.model = self._model(**self.parameters)
        if plot_layers:
            plot_model(self.model, show_shapes=True, to_file="model.png")

    def _model(self, activation_out, optimizer, loss, **kwargs):
        inputs = Input(shape=self.input_shape,
                       name="inputs")

        out_units = self.output_shape[-1]
        out_layer = TimeDistributed(Dense(units=out_units, name="BP_out", activation=activation_out))(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=out_layer,
                               name="BP_model")
        model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
        return model

    def _set_default_parameters(self):
        model_parameters = {"activation_out": "linear"}

        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial):
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "activation_out": activation_out,
        }
        self.parameters.update(model_parameters)

    def get_intermediate_values(self, model):
        return

    def use_intermediate_values(self, inputs, intermediate_function):
        return
