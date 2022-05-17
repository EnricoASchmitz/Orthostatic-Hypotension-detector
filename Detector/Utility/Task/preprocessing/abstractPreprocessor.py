# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Abstract preprocessor implementation

# Imports
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from Detector.Utility.PydanticObject import TagsObject


class Preprocessor(ABC):
    @abstractmethod
    def get_df(self, file) -> Tuple[pd.DataFrame, dict, dict]:
        """ Return dataset

        Returns:
            Dataframe, Markers, dictionary with data info
        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")

    @abstractmethod
    def get_tags(self, file) -> TagsObject:
        """ Get extra tags to add to MLflow

        Returns:
            The ID for our current sample, object with "sample" and "data" key.
            Contains data about our sample and our data

        """
        raise NotImplementedError(f"Abstract class function ({__name__}) not overwritten!")
