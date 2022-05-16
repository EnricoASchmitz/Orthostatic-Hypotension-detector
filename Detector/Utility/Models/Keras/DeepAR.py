# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
import math

import numpy as np
import tensorflow as tf
from numpy.random import normal
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, RepeatVector
from tensorflow.keras.layers import Layer
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.utils.vis_utils import plot_model

from Detector.Utility.Models.Keras.kerasmodel import KerasModel


class DeepAR(KerasModel):
    def __init__(self, input_shape, output_shape, parameters, data_object, plot_layers=False, **kwargs):
        super().__init__()
        disable_eager_execution()

        self._intermediate_layer_name = "gaussian_output"
        self.m_eager = False
        tf.compat.v1.experimental.output_all_intermediates(True)
        self.data_object = data_object
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.set_parameters(parameters)
        self.model = self._architecture(**self.parameters)
        if plot_layers:
            plot_model(self.model, show_shapes=True, to_file="model.png")

    def _architecture(self, lstm_units_architecture, optimizer, **kwargs):
        lstm_units = int(lstm_units_architecture)
        inputs = Input(shape=self.input_shape,
                           name='inputs')

        # First branch: process the sequence
        rnn_out, rnn_state_h, rnn_state_c = LSTM(lstm_units, return_sequences=True,
                                                 name='rnn_layer_architecture', return_state=True)(inputs)

        encoder, enc_state_h, enc_state_c = LSTM(int(lstm_units),
                                                 dropout=0.1,
                                                 return_state=True,
                                                 name="encoder")(rnn_out)
        repeat = RepeatVector(self.output_shape[0], name="future_step_layer")(encoder)
        decoder, dec_state_h, dec_state_c = LSTM(int(lstm_units),
                                                 dropout=0.1, return_sequences=True,
                                                 return_state=True,
                                                 name="decoder")(repeat,
                                                                 initial_state=[enc_state_h,
                                                                                enc_state_c])

        x = TimeDistributed(Dense(int(lstm_units), activation="relu"))(decoder)

        theta = GaussianLayer(self.output_shape[1], name=self._intermediate_layer_name)(x)

        bp_out = theta[0]
        loss = gaussian_likelihood(theta[1])

        model = tf.keras.Model(inputs=inputs, outputs=bp_out,
                               name='BP_model')
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'], run_eagerly=self.m_eager)
        return model

    def _set_default_parameters(self):
        model_parameters = {"lstm_units_architecture": 32}

        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial):
        lstm_units = trial.suggest_int("lstm_units_architecture", 8, 64, step=8)
        model_parameters = {"lstm_units_architecture": lstm_units}
        self.parameters.update(model_parameters)

    def get_intermediate_values(self, model):
        return K.function(
            inputs=[model.input],
            outputs=model.get_layer(self._intermediate_layer_name).output,
        )

    def use_intermediate_values(self, inputs, intermediate_function):
        output = intermediate_function(inputs)
        samples = []
        std = []
        theta = np.array(list(zip(output[0], output[1])))
        for theta_values in theta:
            params, steps, features = theta_values.shape
            sample_steps = []
            std_steps = []
            for timestep in range(steps):
                sample_features = []
                std_features = []
                for feature in range(features):
                    mu, sigma = theta_values[:, timestep, feature]
                    sample = normal(
                        loc=mu, scale=np.sqrt(sigma), size=1
                    )
                    sample_features.append(sample)
                    std_features.append(sigma)
                sample_steps.append(sample_features)
                std_steps.append(std_features)
            samples.append(sample_steps)
            std.append(std_steps)
        samples = np.array(samples)[..., 0]
        std = np.array(std)
        return samples, std

class GaussianLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        """Init."""

        self.output_dim = output_dim
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the weights and biases."""
        n_weight_rows = input_shape[2]
        self.kernel_1 = self.add_weight(
            name="kernel_1",
            shape=(n_weight_rows, self.output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self.kernel_2 = self.add_weight(
            name="kernel_2",
            shape=(n_weight_rows, self.output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self.bias_1 = self.add_weight(
            name="bias_1",
            shape=(self.output_dim,),
            initializer=glorot_normal(),
            trainable=True,
        )
        self.bias_2 = self.add_weight(
            name="bias_2",
            shape=(self.output_dim,),
            initializer=glorot_normal(),
            trainable=True,
        )
        super(GaussianLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """Do the layer computation."""
        output_mu = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        """ Calculate the output dimensions """
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


def gaussian_likelihood(sigma):
    """Likelihood as per the paper."""

    def gaussian_loss(y_true, y_pred):
        """Updated from paper.
        See DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
        """
        loss = tf.reduce_mean(
            tf.math.log(tf.math.sqrt(2 * math.pi))
            + tf.math.log(sigma)
            + tf.math.truediv(
                tf.math.square(y_true - y_pred), 2 * tf.math.square(sigma)
            )
        )
        return tf.abs(loss)

    return gaussian_loss
