# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Contains all enumerators used to create the wanted method

from enum import Enum


class Files(Enum):
    config = "./Config.json"
    tags = "./Castor_export_agesex.csv"


class Parameters(Enum):
    """ Available parameters """
    iterations = 50
    n_trials = 5
    batch_size = 16
    validation_split = 0.2
    time_row_ms = 10
    default_units = 128
    time = 60
    baseline_length = 40
    standing_length = 150
    future_seconds = 150


class MLModelType(Enum):
    """ Available models """
    xgboost = 'xgb'
    Dense = "dense"
    LSTM = 'lstm'
    StackedLSTM = 'stackedlstm'
    biLSTM = 'bilstm'
    StackedBiLSTM = 'stackedbilstm'
    enc_dec_LSTM = 'enc_dec_lstm'
    enc_dec_att_LSTM = 'enc_dec_att_lstm'
    cnn_lstm = 'cnn_lstm'
    nbeats = "nbeats"
    deepar = "deepar"


class PreProcessorMethod(Enum):
    """ Available Preprocessor methods """
    KLOP = 'klop'
