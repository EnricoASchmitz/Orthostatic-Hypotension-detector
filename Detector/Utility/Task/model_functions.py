# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions needed when using the model

# Imports
import time
from logging import Logger
from typing import Tuple, Optional, Any

import mlflow
import numpy as np
from tensorflow import config

from Detector.Utility.Metrics.Losses import Loss
from Detector.Utility.Models.abstractmodel import Model
from Detector.Utility.PydanticObject import InfoObject


def fitting(model: Model, logger: Logger, x_train_inputs: np.ndarray, y_train_outputs: np.ndarray,
            callbacks: Optional[list] = None) -> \
        Tuple[int, float]:
    """ Fit model and track time needed

    Args:
        model: model to fit
        logger: logger
        x_train_inputs: Training input to use
        y_train_outputs: Training output to use
        callbacks: callbacks to use

    Returns:
        number of iterations, time needed for fitting
    """
    logger.info("fitting model")
    start = time.perf_counter()
    n_iterations = model.fit(x_train_inputs, y_train_outputs, callbacks=callbacks)
    return n_iterations, (time.perf_counter() - start)


def predicting(model: Model, logger: Logger, test_set: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """ Predict with model and track time needed

        Args:
            model: model to fit
            logger: logger
            test_set: Testing set to use for fit

        Returns:
            number of iterations, time needed for fitting
        """
    logger.info(f"predicting with model")
    start = time.perf_counter()
    prediction, std = model.predict(test_set)
    return prediction, std, (time.perf_counter() - start)


def fit_and_predict(info_object: InfoObject, logger: Logger,
                    input_values: np.ndarray, output_values: np.ndarray,
                    indexes: Tuple[list, list], model: Model, step: int, scaler: Any,
                                                      rescale_function: callable):
    """ Fit and predict with a model

    Args:
        info_object: configuration information
        logger: logger
        input_values: X values
        output_values: y values
        indexes: indexes for (training, testing)
        model: Model
        step: CV step
        scaler: used scaler for rescaling
        rescale_function: function to do rescaling

    Returns:
        Model, loss_values
    """
    train_index, test_index = indexes

    n_iterations, training_time = fitting(model=model, logger=logger, x_train_inputs=input_values[train_index],
                                          y_train_outputs=output_values[train_index])
    mlflow.log_metric("Training time", training_time, step=step)

    # predict with model
    prediction, std, predict_time = predicting(model=model, logger=logger, test_set=input_values[test_index])

    # scale data
    prediction = rescale_function(prediction, scaler)
    output_values_unscaled = rescale_function(output_values, scaler)

    mlflow.log_metric("Prediction time", predict_time, step=step)

    logger.info(f"plotting prediction from model {info_object.model}")

    # save loss values for CV
    # todo decide if we want this along with the avg
    # get loss values
    loss_values = Loss().get_loss_values(prediction, output_values_unscaled[test_index])
    cv_loss = {}
    for loss_name in loss_values:
        cv_loss[f"cv_{loss_name}"] = loss_values[loss_name]
    # log loss values
    mlflow.log_metrics(cv_loss, step=step)
    return model, loss_values


def check_gpu() -> bool:
    """ Check if a GPU is available

    Returns:
        boolean, gpu available
    """
    use_gpu = False
    gpus = config.list_physical_devices("GPU")
    if gpus:
        use_gpu = True
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
    return use_gpu
