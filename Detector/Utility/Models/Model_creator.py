# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Script to create a model,
# based on https://github.com/OpenSTEF/openstef/blob/main/openstef/model/model_creator.py

# Imports
from typing import Union

from Detector.Utility.Models.Linear_regession import LinearRegressor
from Detector.Utility.Models.XGBoost import XGB
from Detector.Utility.Models.Keras.parameter_models import Cnn, MLP
from Detector.Utility.Models.abstractmodel import Model
from Detector.enums import MLModelType


class ModelCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        MLModelType.Dense: MLP,
        MLModelType.cnn: Cnn,
        MLModelType.xgboost: XGB,
        MLModelType.linearregression: LinearRegressor,
    }

    @staticmethod
    def create_model(model_type: Union[MLModelType, str], **kwargs: any) -> Model:
        """Create a machine learning model based on model type.
        Args:
            model_type (Union[MLModelType, str]): Model type to construct.
            kwargs (any): Optional keyword argument to pass to the model.
        Raises:
            NotImplementedError: When using an invalid model_type.
        Returns:
            Model: model
        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            model_type = MLModelType(model_type)
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                f"No constructor for '{model_type}', "
                f"valid model_types are: {valid_types}"
            ) from e

        return ModelCreator.MODEL_CONSTRUCTORS[model_type](**kwargs)
