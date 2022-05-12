# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Abstract outlier detection methods, with implementations

# Imports
from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


class OutlierDetectionAlgorithm(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame):
        """ Create outlier detection method

        Args:
            df: dataframe to add the results

        Returns:
            dataframe with added results
        """
        raise NotImplementedError("This is the abstract method!")

    @abstractmethod
    def __call__(self, data: pd.DataFrame, col: str):
        """ Perform isolation forest outlier detection

        Args:
            col: used column name
            data: data to use for a prediction

        Returns:
            dataframe with added results
        """
        raise NotImplementedError("This is the abstract method!")


class MyIsolationForest(OutlierDetectionAlgorithm):
    """ isolation forest outlier detection """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.clf = IsolationForest(random_state=0)

    def __call__(self, data: pd.DataFrame, col: str):
        self.clf.fit(data)
        anom_col = f'anomaly_{col}_if'
        self.df[anom_col] = self.clf.predict(data)
        return self.df


class MyKNN(OutlierDetectionAlgorithm):
    """ k-Nearest Neighbors outlier detection """

    def __init__(self, df: pd.DataFrame, k: int = 3):
        self.df = df
        self.knn_model = NearestNeighbors(n_neighbors=k)

    def __call__(self, data: pd.DataFrame, col: str):
        self.knn_model.fit(data)
        # Gather the kth nearest neighbor distance
        distances, indexes = self.knn_model.kneighbors(data)
        avg_dist = pd.Series(distances.mean(axis=1))
        IQR = MyIQR(self.df)
        df = IQR(avg_dist, col, f_name="knn")
        self.df = df
        return self.df


class MyLOF(OutlierDetectionAlgorithm):
    """ LocalOutlierFactor outlier detection """

    def __init__(self, df: pd.DataFrame, k: int = 3):
        self.clf = LocalOutlierFactor(n_neighbors=k)
        self.df = df

    def __call__(self, data: Union[pd.Series, pd.DataFrame], col: str):
        anom_col = f'anomaly_{col}_lof'
        self.df[anom_col] = self.clf.fit_predict(data)

        return self.df


class MyIQR(OutlierDetectionAlgorithm):
    """ IQR outlier detection """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, data: Union[pd.Series, pd.DataFrame], col: str, f_name: str = "iqr"):
        assert (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series))
        if isinstance(data, pd.DataFrame):
            data = data[col]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr_value = q3 - q1
        upper_threshold = q3 + 1.5 * iqr_value
        lower_threshold = q1 - 1.5 * iqr_value
        outlier_locs = ((data > upper_threshold) | (data < lower_threshold))
        outlier_index = self.df.index[np.where(outlier_locs)]

        anom_col = f'anomaly_{col}_{f_name}'
        self.df[anom_col] = 1
        self.df.loc[outlier_index, anom_col] = -1
        return self.df
