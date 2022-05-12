# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Functions needed when using the model

# Imports
import time
from logging import Logger
from typing import Tuple, Optional, Union

import mlflow
import numpy as np
from tensorflow import config

from Detector.Utility.Metrics.Losses import Loss
from Detector.Utility.Models.abstractmodel import Model
from Detector.Utility.Plotting.plotting import plot_prediction
from Detector.Utility.Task.setup_data import inverse_scale


def fitting(model: Model, logger: Logger, train_set: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
            val_set: Union[Tuple[np.ndarray, np.ndarray], np.ndarray], movement_index,
            callbacks: Optional[list] = None) -> \
        Tuple[int, float]:
    """ Fit model and track time needed

    Args:
        model: model to fit
        logger: logger
        train_set: training set to use for fit
        val_set: validation set to use for fit
        callbacks: callbacks to use

    Returns:
        number of iterations, time needed for fitting
    """
    logger.info("fitting model")
    start = time.perf_counter()
    n_iterations = model.fit(train_set=train_set, val_set=val_set, movement_index=movement_index, callbacks=callbacks)
    return n_iterations, (time.perf_counter() - start)


def predicting(model: Model, logger: Logger, test_set: Union[Tuple[np.ndarray, np.ndarray], np.ndarray], movement_index,
               n_out_steps: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """ Predict with model and track time needed

        Args:
            model: model to fit
            logger: logger
            test_set: Testing set to use for fit
            n_out_steps: number of outputs for model

        Returns:
            number of iterations, time needed for fitting
        """
    logger.info(f"predicting with model")
    start = time.perf_counter()
    prediction, std = model.predict(test_set, movement_index=movement_index)
    return prediction, std, (time.perf_counter() - start)


def fit_and_predict(info_object, logger, df, model, n_out_steps, sets, indexes, scaler, step):
    train_set, val_set, test_set = sets
    train_index, movement_index, target_index = indexes

    n_iterations, training_time = fitting(model=model, logger=logger, train_set=train_set, val_set=val_set,
                                          movement_index=movement_index)
    mlflow.log_metric("Training time", training_time, step=step)

    # predict with model
    prediction, std, predict_time = predicting(model=model, logger=logger, test_set=test_set,
                                               n_out_steps=n_out_steps,
                                               movement_index=movement_index)

    mlflow.log_metric("Prediction time", predict_time, step=step)

    logger.info(f"plotting prediction from model {info_object.model}")

    folder_name = f"fitting/cv_{step}/"
    # plot architecture outputs
    fit_X = np.vstack([train_set[0], val_set[0]])
    fit_y = np.vstack([train_set[1], val_set[1]])
    fitting_set = [fit_X, fit_y]
    keys = set(list(train_index.keys()) + list(target_index.keys()))
    remaining_keys = set(list(df.columns)) - keys
    for i, name in enumerate(remaining_keys):
        if std is not None:
            standard_dev = std[..., i]
        else:
            standard_dev = None
        title = f"Prediction {name} ({n_iterations} iterations)".replace("_", " ")
        plot_prediction(target_name=name, target_index=len(target_index) + i, train=fitting_set[1],
                        prediction=prediction,
                        true=test_set[1], title=title, std=standard_dev, folder_name=folder_name)

    # reverse data scaling
    train, test, prediction = inverse_scale(scaler=scaler, train=fitting_set,
                                            test=test_set, prediction=prediction)

    # plot prediction for targets
    for name, column in target_index.items():
        title = f"Prediction {name} ({n_iterations} iterations)".replace("_", " ")
        plot_prediction(target_name=name, target_index=column, train=train, prediction=prediction, true=test,
                        title=title, folder_name=folder_name)

    # save loss values for CV
    # todo decide if we want this along with the avg
    # get loss values
    loss_values = Loss().get_loss_values(prediction, test)
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
    gpus = config.list_physical_devices('GPU')
    if gpus:
        use_gpu = True
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
    return use_gpu
