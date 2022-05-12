# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script:Create the needed preprocessing method

# Imports
from typing import Union

from Detector.Utility.Task.preprocessing.KlopPreprocessor import KlopPreprocessor
from Detector.Utility.Task.preprocessing.abstractPreprocessor import Preprocessor
from Detector.enums import PreProcessorMethod


class PreprocessorCreator:
    """Factory object for creating the correct preprocessor"""

    # Set object mapping
    Preprocessor_CONSTRUCTORS = {
        PreProcessorMethod.KLOP: KlopPreprocessor
    }

    @staticmethod
    def get_preprocessor(method: Union[PreProcessorMethod, str], **kwargs: any) -> Preprocessor:
        """Create a preprocessing method based on method.
        Args:
            method (Union[PreProcessorMethod, str]): Model type to construct.
            kwargs (any): Optional keyword argument to pass to the model.
        Raises:
            NotImplementedError: When using an invalid method.
        Returns:
            Preprocessor: Preprocessor method
        """
        try:
            # This will raise a ValueError when an invalid method str is used
            # and nothing when a PreProcessorMethod enum is used.
            method = PreProcessorMethod(method)
        except ValueError as e:
            valid_types = [t.value for t in PreProcessorMethod]
            raise NotImplementedError(
                f"No constructor for '{method}', "
                f"valid preprocess methods are: {valid_types}"
            ) from e

        return PreprocessorCreator.Preprocessor_CONSTRUCTORS[method](**kwargs)
