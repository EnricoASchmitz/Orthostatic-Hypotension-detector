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
    iterations = 100
    n_trials = 20
    batch_size = 28
    avg_bpm = 80
    minutes = 3
    time_row_ms = 50
    default_units = 128
    shifts = [
                 0.5,
                 1,
                 3,
                 5,
                 10
             ],


class MLModelType(Enum):
    """ Available models """
    Baseline = "baseline"
    LSTM = 'lstm'
    StackedLSTM = 'stackedlstm'
    biLSTM = 'bilstm'
    StackedBiLSTM = 'stackedbilstm'
    enc_dec_LSTM = 'enc_dec_lstm'
    enc_dec_att_LSTM = 'enc_dec_att_lstm'
    cnn = 'cnn'
    xgboost = 'xgb'


class ODAlgorithm(Enum):
    """ Available Outlier detection methods """
    KNN = "knn"
    LOF = "lof"
    IF = "if"
    IQR = "iqr"


class PreProcessorMethod(Enum):
    """ Available Preprocessor methods """
    KLOP = 'klop'
    NILVAD = 'nilvad'


class Architectures(Enum):
    """ Available architectures """
    basic = "basic"
    nbeats = "nbeats"
    deepAR = "deepar"
