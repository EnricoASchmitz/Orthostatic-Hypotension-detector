# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports

from nbeats_keras.model import NBeatsNet as NBeatsKeras

from Detector.Utility.Models.Keras.Architecture import Architecture
from Detector.enums import Parameters


class NBeatsArchitecture(Architecture):
    def __init__(self, n_in_steps, n_in_features, n_out_steps, data_object):
        super().__init__(n_in_steps, n_in_features, n_out_steps, data_object)

    def __call__(self, nb_blocks_per_stack, units_architecture, generic_dim, trend_dim, seasonality_dim, **kwargs):
        # nbeats part
        stack_types = []
        thetas_dim = []
        generic_dim = int(generic_dim)
        trend_dim = int(trend_dim)
        seasonality_dim = int(seasonality_dim)
        if generic_dim > 0:
            stack_types.append("generic")
            thetas_dim.append(generic_dim)
        if trend_dim > 0:
            stack_types.append("trend")
            thetas_dim.append(trend_dim)
        if seasonality_dim > 0:
            stack_types.append("seasonality")
            thetas_dim.append(seasonality_dim)
        movement_features = len(self.data_object.movement_features)
        nbeats = NBeatsKeras(input_dim=self.n_in_features,  # oxy, dxy
                             output_dim=self.n_in_features,  # oxy, dxy
                             exo_dim=movement_features,  # for the movement branch
                             forecast_length=self.n_out_steps,  # future time steps
                             backcast_length=self.n_in_steps,  # past time steps
                             stack_types=stack_types,
                             # different stacks to use ("generic", "trend", "seasonality")
                             thetas_dim=thetas_dim,  # need to match with stack types len
                             nb_blocks_per_stack=int(nb_blocks_per_stack),
                             hidden_layer_units=int(units_architecture)
                             )
        inputs = nbeats.input
        oxy_out = nbeats.output
        return inputs, oxy_out, None

    def get_parameters(self):
        return {"nb_blocks_per_stack": 3,
                "generic_dim": 0,
                "trend_dim": 4,
                "seasonality_dim": 8,
                "units_architecture": int(Parameters.default_units.value)
                }

    def get_trial_parameters(self, trial):
        nb_blocks = trial.suggest_int("nb_blocks_per_stack", 1, 3)
        units = trial.suggest_int("units_architecture", 32, int(Parameters.default_units.value * 2), step=32)
        generic_dim = trial.suggest_int("generic_dim", 0, 1)
        trend_dim = trial.suggest_int("trend_dim", 0, 6)
        seasonality_dim = trial.suggest_int("seasonality_dim", 0, 10)
        architecture_parameters = {
            "nb_blocks_per_stack": nb_blocks,
            "units_architecture": units,
            "generic_dim": generic_dim,
            "trend_dim": trend_dim,
            "seasonality_dim": seasonality_dim
        }

        return architecture_parameters

    def get_intermediate_values(self, model):
        return

    def use_intermediate_values(self, sample, intermediate_function):
        return
