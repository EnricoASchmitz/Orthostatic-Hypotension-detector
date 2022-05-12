# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
import optuna

from Detector.Utility.Models.Keras.kerasmodel import Base


class Baseline(Base):
    def __init__(self, data_object, n_in_steps, n_out_steps: int, n_in_features: int, n_mov_features: int,
                 gpu, plot_layers=False,
                 parameters=None):
        """ Create BiLSTM model

        :param n_in_steps: Number of time steps to look back
        :param n_features: Number of futures
        :param gpu: if GPU is available
        """
        super().__init__(data_object=data_object,
                         n_in_steps=n_in_steps,
                         n_out_steps=n_out_steps,
                         n_in_features=n_in_features,
                         n_mov_features=n_mov_features,
                         gpu=gpu,
                         plot_layers=plot_layers,
                         parameters=parameters
                         )

    def _get_model(self):
        return None

    def _set_default_parameters(self):
        # contains no mapper, so no new parameters
        pass

    def _set_optuna_parameters(self, trial: optuna.Trial):
        # contains no mapper, so no new parameters
        pass
