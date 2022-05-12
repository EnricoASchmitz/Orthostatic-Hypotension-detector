# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Get the specified outlier detection method

# Imports
from typing import Union

from Detector.Utility.Data_preprocessing.Outlier_detection.OD_algorithms import MyKNN, MyIsolationForest, MyIQR, \
    MyLOF, OutlierDetectionAlgorithm
from Detector.enums import ODAlgorithm


class OutlierDetector:
    """Factory object for creating outlier detection method"""
    # Set object mapping
    OD_CONSTRUCTORS = {
        ODAlgorithm.KNN: MyKNN,
        ODAlgorithm.IF: MyIsolationForest,
        ODAlgorithm.IQR: MyIQR,
        ODAlgorithm.LOF: MyLOF,
    }

    @staticmethod
    def create_algorithm(od_type: Union[ODAlgorithm, str], **kwargs: any) -> OutlierDetectionAlgorithm:
        """Create an outlier detection method on od_type.
        Args:
            od_type: outlier detection method to construct.
            kwargs: Optional keyword argument to pass to the model.
        Raises:
            NotImplementedError: When using an invalid model_type.
        Returns:
            OutlierDetectionAlgorithm: outlier detection method
        """
        try:
            # This will raise a ValueError when an invalid od_type str is used
            # and nothing when a ODAlgorithm enum is used.
            od_type = ODAlgorithm(od_type)
        except ValueError as e:
            valid_types = [t.value for t in ODAlgorithm]
            raise NotImplementedError(
                f"No constructor for '{od_type}', "
                f"valid od_types are: {valid_types}"
            ) from e

        return OutlierDetector.OD_CONSTRUCTORS[od_type](**kwargs)
