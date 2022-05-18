# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras CNN implementation

# Imports
from keras.layers import LSTM
from optuna import Trial
from tensorflow.keras.layers import Conv1D

from Detector.Utility.Models.Keras.kerasmodel import Base
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


class CNN_LSTM(Base):
    """ Basic CNN model """

    # todo fix Call to CreateProcess failed. Error code: 2
    def __init__(self, data_object: DataObject, input_shape,
                 output_shape, gpu: bool, plot_layers=False,
                 parameters=None):
        """ Create CNN model

        :param n_in_steps: Number of time steps to look back
        :param n_features: Number of futures
        :param gpu: if GPU is available
        """
        super().__init__(data_object=data_object,
                         input_shape=input_shape,
                         output_shape=output_shape,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def cnn_mapper(self, input_layer, units_mapper, kernel_size, activation, dropout=0.0, **kwargs):
        cnn_layer = Conv1D(filters=int(units_mapper), kernel_size=int(kernel_size), padding="same",
                           activation=activation
                           )(input_layer)
        lstm_layer = LSTM(int(units_mapper), dropout=float(dropout))(cnn_layer)
        out_layer = self._output_layers_parameters(lstm_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.cnn_mapper

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "kernel_size": 5,
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: Trial):
        # get CNN parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=32)
        kernel_size = trial.suggest_int("kernel_size", 2, 6, step=1)
        model_parameters = {
            "units_mapper": units_mapper,
            "kernel_size": kernel_size,
        }
        self.parameters.update(model_parameters)
