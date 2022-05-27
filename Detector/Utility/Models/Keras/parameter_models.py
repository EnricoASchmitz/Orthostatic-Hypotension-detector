# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras parameter model implementations

# Imports

import optuna
from tensorflow.keras.layers import LSTM, Bidirectional, Reshape, Conv1D, Attention, Concatenate
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

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
            "dropout_value": 0.0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get Dense parameters
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        if n_dense_layers > 0:
            dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        else:
            dropout = 0
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
        }
        self.parameters.update(model_parameters)


class SimpleLSTM(Base):
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

    def lstm_layer(self, input_layer, units_layer, dropout, **kwargs):
        lstm_layer = LSTM(int(units_layer), dropout=float(dropout))(input_layer)
        bp_out = self._output_layers_parameters(lstm_layer, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.lstm_layer

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "n_dense_layers": 0,
            "dropout": 0.0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
        }
        self.parameters.update(model_parameters)


class BiLSTM(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create BiLSTM model

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

    def bilstm_layer(self, input_layer, units_layer, dropout, **kwargs):
        bilstm_layer = Bidirectional(LSTM(int(units_layer), dropout=float(dropout)))(
            input_layer)
        bp_out = self._output_layers_parameters(bilstm_layer, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.bilstm_layer

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "dropout": 0.0,
            "n_dense_layers": 0,
            "dropout_value": 0.0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
        }
        self.parameters.update(model_parameters)


class StackedLSTM(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create StackedLSTM model

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

    def recursive_block(self, prev_layer: KerasTensor, n_blocks: int, units_layer: int, dropout: float, **kwargs):
        """ Create a stacked bilstm-lstm block

        Args:
            prev_layer: Layer to append block to
            n_blocks: number of blocks
            units_layer: number of hidden unit's
            dropout: dropout used in LSTM

        Returns:
            Last layer in the block
        """
        lstm_nodes = int(units_layer)
        dropout = float(dropout)
        for layer in range(int(n_blocks)):
            if layer == int(n_blocks) - 1:
                prev_layer = LSTM(lstm_nodes, dropout=dropout)(prev_layer)
            else:
                prev_layer = LSTM(lstm_nodes, dropout=dropout, return_sequences=True)(prev_layer)
        bp_out = self._output_layers_parameters(prev_layer, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.recursive_block

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "n_blocks": 2,
            "dropout": 0.0,
            "n_dense_layers": 0,
            "dropout_value": 0.0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=16)
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
            "n_blocks": n_blocks
        }
        self.parameters.update(model_parameters)


class StackedBiLSTM(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create Stacked BiLSTM model

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

    def recursive_block(self, prev_layer: KerasTensor, n_blocks: int, units_layer: int, dropout: float, **kwargs) -> KerasTensor:
        """ Create a stacked bilstm-lstm block

        Args:
            prev_layer: Layer to append block to
            n_blocks: number of blocks
            units_layer: number of hidden unit's
            dropout: dropout used in LSTM

        Returns:
            Last layer in the block
        """
        bilstm_nodes = int(units_layer)
        dropout = float(dropout)
        for layer in range(int(n_blocks)):
            if layer == int(n_blocks) - 1:
                prev_layer = LSTM(bilstm_nodes, dropout=dropout)(prev_layer)
            else:
                prev_layer = LSTM(bilstm_nodes, dropout=dropout, return_sequences=True)(prev_layer)
        bp_out = self._output_layers_parameters(prev_layer, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.recursive_block

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "n_blocks": 2,
            "dropout": 0.0,
            "n_dense_layers": 0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=16)
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
            "n_blocks": n_blocks
        }
        self.parameters.update(model_parameters)


class EncDecLSTM(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create EncDecLSTM model

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

    def encdec_layer(self, input_layer: KerasTensor, units_layer: int, dropout: float, **kwargs):
        units_layer = int(units_layer)
        dropout = float(dropout)
        bp_encoder, enc_state_h, enc_state_c = LSTM(units_layer, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_encoder")(input_layer)
        bp_decoder, dec_state_h, dec_state_c = LSTM(units_layer, dropout=dropout,
                                                           return_state=True,
                                                           name="translate_decoder")(bp_encoder,
                                                                                     initial_state=[enc_state_h,
                                                                                                    enc_state_c])
        bp_out = self._output_layers_parameters(bp_decoder, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.encdec_layer

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "dropout": 0.0,
            "n_dense_layers": 0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
        }
        self.parameters.update(model_parameters)


class EncDecAttLSTM(Base):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 gpu, plot_layers=False, parameters=None):
        """ Create EncDecAttLSTM model

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

    def encdecatt_layer(self, input_layer: KerasTensor, units_layer: int, dropout: float, **kwargs):
        units_layer = int(units_layer)
        dropout = float(dropout)
        bp_encoder, enc_state_h, enc_state_c = LSTM(units_layer, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_encoder")(input_layer)
        bp_decoder, dec_state_h, dec_state_c = LSTM(units_layer, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_decoder")(bp_encoder,
                                                                                     initial_state=[enc_state_h,
                                                                                                    enc_state_c])
        attn_out = Attention(name='attention_layer')([bp_encoder, bp_decoder])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([bp_decoder, attn_out])
        lstm_layer = LSTM(int(units_layer), dropout=float(dropout))(decoder_concat_input)
        bp_out = self._output_layers_parameters(lstm_layer, dropout=dropout, **kwargs)
        return bp_out

    def _get_model(self):
        return self.encdecatt_layer

    def _set_default_parameters(self):
        model_parameters = {
            "units_layer": int(Parameters.default_units.value),
            "dropout": 0.0,
            "n_dense_layers": 0,
            "activation_out": "linear",
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
        }
        self.parameters.update(model_parameters)


class CnnLSTM(Base):
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

    def cnn_layer(self, input_layer, units_layer, kernel_size, dropout=0.0, **kwargs):
        cnn_layer = Conv1D(filters=int(units_layer), kernel_size=int(kernel_size), padding="same")(input_layer)
        lstm_layer = LSTM(int(units_layer), dropout=float(dropout))(cnn_layer)
        bp_out = self._output_layers_parameters(lstm_layer, dropout=dropout, **kwargs)
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
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get CNN parameters
        units_layer = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        kernel_size = trial.suggest_int("kernel_size", 2, 6, step=1)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        activation_out = trial.suggest_categorical("activation_out", ["tanh", "linear"])
        model_parameters = {
            "n_dense_layers": n_dense_layers,
            "activation_out": activation_out,
            "dropout": dropout,
            "units_layer": units_layer,
            "kernel_size": kernel_size,
        }
        self.parameters.update(model_parameters)
