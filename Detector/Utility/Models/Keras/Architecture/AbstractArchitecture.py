# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from optuna import Trial
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


class Architecture(ABC):
    def __init__(self, n_in_steps, n_in_features, n_out_steps, data_object):
        self.sep_mapper = None
        self.n_in_steps = n_in_steps
        self.n_in_features = n_in_features
        self.n_out_steps = n_out_steps
        self.data_object = data_object
        self.AR_eager = True

    def __call__(self, **kwargs) -> Tuple[KerasTensor, KerasTensor, Optional[callable]]:
        raise NotImplementedError("Abstract class not overwritten")

    @abstractmethod
    def get_parameters(self) -> dict:
        raise NotImplementedError("Abstract class not overwritten")

    @abstractmethod
    def get_trial_parameters(self, trial: Trial) -> dict:
        raise NotImplementedError("Abstract class not overwritten")

    @abstractmethod
    def get_intermediate_values(self, model):
        return None

    @abstractmethod
    def use_intermediate_values(self, sample, intermediate_function):
        return None, None
