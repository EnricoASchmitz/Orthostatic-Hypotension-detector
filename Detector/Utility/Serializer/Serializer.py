# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Saving and loading data, Serializer

# Imports
import os
from time import sleep
from typing import Optional

import mlflow
import pandas as pd


class MLflowSerializer:
    """ Save information to MLflow """

    def __init__(self, dataset_name: str, parameter_expiriment: bool, sample_tags: dict):
        # setup
        self.uri_start = "file://"
        pwd = os.getcwd()
        if pwd[0] != "/":
            self.uri_start = f"{self.uri_start}/"
        tracking_uri = f"{self.uri_start}{pwd}/{dataset_name}"
        mlflow.set_tracking_uri(tracking_uri)
        sample_tags.update({"dataset": dataset_name})
        if parameter_expiriment:
            exp_name = "Parameters"
        else:
            exp_name = "Full curve"
        self.experiment_id = self._set_experiment(exp_name, sample_tags)

    def _set_experiment(self, exp_name, sample_tags):
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            mlflow.set_experiment(exp_name)
            experiment_id = experiment.experiment_id
        else:
            try:
                experiment_id = mlflow.create_experiment(exp_name, tags=sample_tags)
            except FileExistsError:
                sleep(10)
                experiment_id = self._set_experiment(exp_name, sample_tags)
        return experiment_id

    def get_last_optimized_run(self, name: str) -> Optional[pd.Series]:
        """ Return the last optimized run

        Args:
            name: the run name, is the model name

        Returns:
            if an old run exists, we return the latest optimization run
        """
        # use mlflow to search old runs, returns a pd.
        filter_run_name = f"tags.mlflow.runName = '{name}'"
        filter_phase = "tags.phase = 'Optimizing'"
        last_run = mlflow.search_runs([self.experiment_id],
                                      filter_string=f"{filter_phase} AND {filter_run_name}",
                                      max_results=1)
        # if there are no runs we remove the old file and start clean
        if last_run.empty:
            return
        # only return the latest run
        return last_run.iloc[0]

    def get_last_training_run(self, name: str) -> Optional[pd.Series]:
        """ Return the last optimized run

        Args:
            name: the run name, is the model name

        Returns:
            if an old run exists, we return the latest optimization run
        """
        # use mlflow to search old runs, returns a pd.
        filter_run_name = f"tags.mlflow.runName = '{name}'"
        filter_phase = "tags.phase = 'training'"
        query = f"{filter_phase} and {filter_run_name}"
        last_run = mlflow.search_runs([self.experiment_id],
                                      filter_string=query,
                                      max_results=1)
        # if there are no runs we remove the old file and start clean
        if last_run.empty:
            return
        # only return the latest run
        return last_run.iloc[0]
