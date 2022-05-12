# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
from keras import Input
from keras.layers import LSTM, TimeDistributed, Dense, Add, RepeatVector, Lambda

from Detector.Utility.Models.Keras.Architecture import Architecture
from Detector.enums import Parameters


class BasicArchitecture(Architecture):
    def __init__(self, n_in_steps, n_in_features, n_out_steps, data_object):
        super().__init__(n_in_steps, n_in_features, n_out_steps, data_object)

    def __call__(self, lstm_units_architecture, **kwargs):
        lstm_units = int(lstm_units_architecture)
        oxy_dxy_in = Input(shape=(self.n_in_steps, self.n_in_features),
                           name='oxy_dxy_in')

        # First branch: process the sequence
        rnn_out, rnn_state_h, rnn_state_c = LSTM(lstm_units, return_sequences=True,
                                                 name='rnn_layer', return_state=True)(oxy_dxy_in)
        # Need because the rnn_out can't be passed directly to add, type of output not correct, this layer does nothing
        pref_layer = Lambda(passing, name="empty_layer")(rnn_out)

        # Second branch:capture movement
        movement_features = len(self.data_object.movement_features)
        if movement_features > 0:
            mov_in = Input(shape=(self.n_in_steps, movement_features),
                           name='mov_in')
            # Dense to 1 variable
            if movement_features >= 2:
                dense_in = TimeDistributed(Dense(1, activation="sigmoid"))(mov_in)
                add = [pref_layer, dense_in]
            else:
                add = [pref_layer, mov_in]
            # Combined output: add the two branches
            pref_layer = TimeDistributed(Add(name='add_movement_oxy'), name="time_mov_oxy")(add)

            inputs = [oxy_dxy_in, mov_in]
        else:
            inputs = oxy_dxy_in

        # oxy
        # encoder decoder
        encoder, state_h, state_c = LSTM(lstm_units,
                                         return_sequences=False,
                                         return_state=True,
                                         name="oxy_encoder"
                                         )(pref_layer, initial_state=[rnn_state_h, rnn_state_c])

        repeat = RepeatVector(self.n_out_steps, name="future_step_layer")(encoder)
        decoder = LSTM(lstm_units, return_sequences=True, name="oxy_decoder")(repeat,
                                                                              initial_state=[state_h, state_c])
        oxy_out = TimeDistributed(Dense(self.n_in_features, name='oxy_out'), name="steps_oxy")(decoder)
        return inputs, oxy_out, None

    def get_parameters(self):
        return {"lstm_units_architecture": int(Parameters.default_units.value)}

    def get_trial_parameters(self, trial):
        lstm_units = trial.suggest_int("lstm_units_architecture", 32, int(Parameters.default_units.value * 2), step=32)
        return {"lstm_units_architecture": lstm_units}

    def get_intermediate_values(self, model):
        return

    def use_intermediate_values(self, sample, intermediate_function):
        return


def passing(tensor):
    return tensor
