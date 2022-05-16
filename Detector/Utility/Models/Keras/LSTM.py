# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras LSTM implementations

# Imports

import optuna
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.python.layers.base import Layer

from Detector.Utility.Models.Keras.kerasmodel import Base
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


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

    def lstm_mapper(self, input_layer, units_mapper, dropout, **kwargs):
        lstm_layer = LSTM(int(units_mapper), dropout=float(dropout))(input_layer)
        out_layer = self._output_layers_parameters(lstm_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.lstm_mapper

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "dropout": 0.0
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
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

    def bilstm_mapper(self, input_layer, units_mapper, dropout, **kwargs):
        bilstm_layer = Bidirectional(LSTM(int(units_mapper), dropout=float(dropout)))(
            input_layer)
        out_layer = self._output_layers_parameters(bilstm_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.bilstm_mapper

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "dropout": 0.0
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
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

    def recursive_block(self, prev_layer: layers, n_blocks: int, units_mapper: int, dropout: float, **kwargs):
        """ Create a stacked bilstm-lstm block

        Args:
            prev_layer: Layer to append block to
            n_blocks: number of blocks
            units_mapper: number of hidden units
            dropout: dropout used in LSTM

        Returns:
            Last layer in the block
        """
        lstm_nodes = int(units_mapper)
        dropout = float(dropout)
        for layer in range(int(n_blocks)):
            if layer == int(n_blocks) - 1:
                prev_layer = LSTM(lstm_nodes, dropout=dropout, activation='tanh')(prev_layer)
            else:
                prev_layer = LSTM(lstm_nodes, dropout=dropout, activation='tanh', return_sequences=True)(prev_layer)
        out_layer = self._output_layers_parameters(prev_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.recursive_block

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "n_blocks": 2,
            "dropout": 0.0,
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
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

    def recursive_block(self, prev_layer: layers, n_blocks: int, units_mapper: int, dropout: float, **kwargs) -> Layer:
        """ Create a stacked bilstm-lstm block

        Args:
            prev_layer: Layer to append block to
            n_blocks: number of blocks
            units_mapper: number of hidden units
            dropout: dropout used in LSTM

        Returns:
            Last layer in the block
        """
        bilstm_nodes = int(units_mapper)
        dropout = float(dropout)
        for layer in range(int(n_blocks)):
            if layer == int(n_blocks) - 1:
                prev_layer = LSTM(bilstm_nodes, dropout=dropout, activation='tanh')(prev_layer)
            else:
                prev_layer = LSTM(bilstm_nodes, dropout=dropout, activation='tanh', return_sequences=True)(prev_layer)
        out_layer = self._output_layers_parameters(prev_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.recursive_block

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "n_blocks": 2,
            "dropout": 0.0,
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
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

    def encdec_mapper(self, input_layer, units_mapper, dropout, **kwargs):
        units_mapper = int(units_mapper)
        dropout = float(dropout)
        bp_encoder, enc_state_h, enc_state_c = layers.LSTM(units_mapper, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_encoder")(input_layer)
        bp_decoder, dec_state_h, dec_state_c = layers.LSTM(units_mapper, dropout=dropout,
                                                           return_state=True,
                                                           name="translate_decoder")(bp_encoder,
                                                                                     initial_state=[enc_state_h,
                                                                                                    enc_state_c])
        out_layer = self._output_layers_parameters(bp_decoder, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.encdec_mapper

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "dropout": 0.0
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
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

    def encdecatt_mapper(self, input_layer, units_mapper, dropout, **kwargs):
        units_mapper = int(units_mapper)
        dropout = float(dropout)
        bp_encoder, enc_state_h, enc_state_c = layers.LSTM(units_mapper, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_encoder")(input_layer)
        bp_decoder, dec_state_h, dec_state_c = layers.LSTM(units_mapper, dropout=dropout, return_sequences=True,
                                                           return_state=True,
                                                           name="translate_decoder")(bp_encoder,
                                                                                     initial_state=[enc_state_h,
                                                                                                    enc_state_c])
        attn_out = layers.Attention(name='attention_layer')([bp_encoder, bp_decoder])
        decoder_concat_input = layers.Concatenate(axis=-1, name='concat_layer')([bp_decoder, attn_out])
        lstm_layer = LSTM(int(units_mapper), dropout=float(dropout))(decoder_concat_input)
        out_layer = self._output_layers_parameters(lstm_layer, dropout_value=float(dropout), activation="tanh")
        return out_layer

    def _get_model(self):
        return self.encdecatt_mapper

    def _set_default_parameters(self):
        model_parameters = {
            "units_mapper": int(Parameters.default_units.value),
            "dropout": 0.0
        }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # get LSTM parameters
        units_mapper = trial.suggest_int("units_mapper", 32, int(Parameters.default_units.value * 2), step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        model_parameters = {
            "units_mapper": units_mapper,
            "dropout": dropout,
        }
        self.parameters.update(model_parameters)
