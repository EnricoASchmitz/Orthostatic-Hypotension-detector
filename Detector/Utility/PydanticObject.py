# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Pydantic object models to assure, all needed data is specified

# Imports
from typing import List, Optional

from pydantic import BaseModel

from Detector.enums import Parameters


class InfoObject(BaseModel):
    """ Object to save the needed information about the setup, taken from config """
    dataset: str
    input_steps: int
    output_steps: int
    file_loc: str
    smooth: bool
    model: str
    outlier_algo: str
    time_row: Optional[int] = Parameters.time_row_ms.value
    shifts: Optional[List[int]] = Parameters.shifts.value


class DataObject(BaseModel):
    """ Object to save the needed information about the data """
    index_col: str
    nirs_col: List[str]
    target_col: List[str]
    features: List[str]
    hz: int
    train_features: Optional[List[str]] = []
    movement_features: Optional[List[str]] = []
    tags: Optional[dict] = None
    rows_per_beat: Optional[float] = None
    reindex: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_rows_per_beat(self) -> float:
        """  Calculate the numbers of rows within one heartbeat. Safe in the object.

        :return: number of rows in each beat
        """
        # convert bpm to bps
        avg_bps = Parameters.avg_bpm.value / 60

        # divide the number of rows in a second with the bps to get the rows for each beat
        self.rows_per_beat = self.hz / avg_bps
        return self.rows_per_beat


class TagsObject(BaseModel):
    """ Data to add to MLflow """
    ID: str
    sample: dict
