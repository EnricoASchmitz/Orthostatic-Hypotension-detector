# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: 

# Imports
from typing import Union

from Detector.Utility.Models.Keras.Architecture.AbstractArchitecture import Architecture
from Detector.Utility.Models.Keras.Architecture.DeepAR import DeepARArchitecture
from Detector.Utility.Models.Keras.Architecture.basic import BasicArchitecture
from Detector.Utility.Models.Keras.Architecture.nbeats import NBeatsArchitecture
from Detector.enums import Architectures


class ArchitectureCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    Architecture_CONSTRUCTORS = {
        Architectures.basic: BasicArchitecture,
        Architectures.nbeats: NBeatsArchitecture,
        Architectures.deepAR: DeepARArchitecture,
    }

    @staticmethod
    def create_architecture(architecture_type: Union[Architectures, str], **kwargs: any) -> Architecture:
        """Create a basic architecture.
        Args:
            architecture_type (Union[Architectures, str]): Model type to construct.
            kwargs (any): Optional keyword argument to pass to the model.
        Raises:
            NotImplementedError: When using an invalid architecture_type.
        Returns:
            Architecture: architecture
        """
        try:
            # This will raise a ValueError when an invalid architecture_type str is used
            # and nothing when a Architectures enum is used.
            architecture_type = Architectures(architecture_type)
        except ValueError as e:
            valid_types = [t.value for t in architecture_type]
            raise NotImplementedError(
                f"No constructor for '{architecture_type}', "
                f"valid architecture_type are: {valid_types}"
            ) from e

        return ArchitectureCreator.Architecture_CONSTRUCTORS[architecture_type](**kwargs)
