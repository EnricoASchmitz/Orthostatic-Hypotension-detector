# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Pydantic object models to assure, all needed data is specified

# Imports
from typing import List, Optional, Any

from pydantic import BaseModel

from Detector.enums import Parameters


class InfoObject(BaseModel):
    """ Object to save the needed information about the setup, taken from config """
    dataset: str
    file_loc: str
    smooth: bool
    model: str
    nirs_input: bool
    parameter_model: bool = None
    time_row: Optional[int] = Parameters.time_row_ms.value


class DataObject(BaseModel):
    """ Object to save the needed information about the data """
    index_col: str
    nirs_col: List[str]
    target_col: List[str]
    hz: int
    reindex: bool = False
    scaler: Any = None


class TagsObject(BaseModel):
    """ Data to add to MLflow """
    ID: str
    sample: dict
