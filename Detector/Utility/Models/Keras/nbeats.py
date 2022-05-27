# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports

from nbeats_keras.model import NBeatsNet as NBeatsKeras
from tensorflow.python.keras.utils.vis_utils import plot_model

from Detector.Utility.Models.Keras.kerasmodel import KerasModel
from Detector.Utility.PydanticObject import DataObject
from Detector.enums import Parameters


class NBeats(KerasModel):
    def __init__(self, data_object: DataObject, input_shape, output_shape,
                 plot_layers=False, parameters=None, **kwargs):
        super().__init__()
        self.data_object = data_object
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.set_parameters(parameters)
        self.model = self._model(**self.parameters)
        if plot_layers:
            plot_model(self.model, show_shapes=True, to_file="model.png")

    def _model(self, nb_blocks_per_stack, units_layer,
               generic_dim, trend_dim, seasonality_dim,
               optimizer, loss, **kwargs):
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
        # todo: gave key error
        nbeats = NBeatsKeras(input_dim=self.input_shape[-1],  # oxy, dxy
                             output_dim=self.output_shape[-1],  # oxy, dxy
                             forecast_length=self.input_shape[0],  # future time steps
                             backcast_length=self.output_shape[0],  # past time steps
                             stack_types=stack_types,
                             # different stacks to use ("generic", "trend", "seasonality")
                             thetas_dim=thetas_dim,  # need to match with stack types len
                             nb_blocks_per_stack=int(nb_blocks_per_stack),
                             hidden_layer_units=int(units_layer)
                             )
        nbeats.compile(optimizer=optimizer, loss=loss, metrics=['mae'], run_eagerly=self.m_eager)
        return nbeats

    def _set_default_parameters(self):
        model_parameters = {"nb_blocks_per_stack": 3,
                            "generic_dim": 0,
                            "trend_dim": 4,
                            "seasonality_dim": 8,
                            "units_layer": int(Parameters.default_units.value)
                            }
        self.parameters.update(model_parameters)

    def _set_optuna_parameters(self, trial):
        nb_blocks = trial.suggest_int("nb_blocks_per_stack", 1, 3)
        units = trial.suggest_int("units_layer", 32, int(Parameters.default_units.value * 2), step=32)
        generic_dim = trial.suggest_int("generic_dim", 0, 1)
        trend_dim = trial.suggest_int("trend_dim", 0, 6)
        seasonality_dim = trial.suggest_int("seasonality_dim", 0, 10)
        model_parameters = {
            "nb_blocks_per_stack": nb_blocks,
            "units_layer": units,
            "generic_dim": generic_dim,
            "trend_dim": trend_dim,
            "seasonality_dim": seasonality_dim
        }
        self.parameters.update(model_parameters)

    def get_intermediate_values(self, model):
        return

    def use_intermediate_values(self, sample, intermediate_function):
        return
