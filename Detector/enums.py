# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Contains all enumerators used to create the wanted method

from enum import Enum


class Files(Enum):
    config = "/home/eschmitz/Orthostatic-Hypotension-detector/Config.json"
    tags = "/home/eschmitz/Orthostatic-Hypotension-detector/Castor_export_agesex.csv"


class Parameters(Enum):
    """ Available parameters """
    iterations = 1000
    n_trials = 500
    batch_size = 8
    validation_split = 0.2
    time_row_ms = 10
    default_units = 128
    time = 60
    baseline_tuple = (40, 10)
    rest_length = 40
    standing_length = 150
    future_seconds = 150
    recovery_times = [15, 20, 30, 40, 50, 60, 120, 150]
    loss = "mse"
    minimal_BP = 20
    maximal_BP = 250


class MLModelType(Enum):
    """ Available models """
    xgboost = "xgb"
    Dense = "dense"
    linearregression = "linearregression"
    cnn = "cnn"
    timeMLP = "timemlp"


class PreProcessorMethod(Enum):
    """ Available Preprocessor methods """
    KLOP = "klop"
