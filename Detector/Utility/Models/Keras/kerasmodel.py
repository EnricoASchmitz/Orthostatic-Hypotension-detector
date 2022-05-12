# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Keras general model implementation

# Imports
import gc
import os
from abc import abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Dense, TimeDistributed
from optuna import Trial
from sklearn.model_selection import train_test_split
from tensorflow import keras, get_logger
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tqdm import tqdm

from Detector.Utility.Models.Keras.Architecture import ArchitectureCreator
from Detector.Utility.Models.abstractmodel import Model, get_movement
from Detector.Utility.PydanticObject import DataObject
from Detector.Utility.Task.setup_data import split_series
from Detector.enums import Parameters, Architectures


class KerasModel(Model):
    def __init__(self):
        super().__init__()

        self.get_intermediate = None
        self.architecture = None
        self.model = None
        self.mapper = None
        self.data_object = None
        self.n_out_steps = None
        self.n_in_features = None
        self.n_mov_features = None
        self.n_in_steps = None
        self.fig = True

    def get_data(self, data: np.ndarray, train_index, target_index: dict, val_set: bool, test_set: bool):
        timeserie_X, timeserie_y = split_series(data, self.n_in_steps, self.n_out_steps, train_index, target_index)
        if val_set or test_set:
            data_X = self._split_data(timeserie_X, val_set, test_set)
            data_y = self._split_data(timeserie_y, val_set, test_set)
            data_output = []
            for set_i in range(len(data_X)):
                data_output.append((data_X[set_i], data_y[set_i]))

            return data_output
        else:
            return timeserie_X, timeserie_y

    def _split_data(self, timeserie, val_set, test_set):
        datasets = []
        train_set = None
        if test_set:
            train_set, test_set = train_test_split(timeserie, test_size=0.2, random_state=1, shuffle=False)
            test_start = train_set[-self.n_in_steps:]
            test_with_start = np.concatenate((test_start, test_set), axis=0)
            datasets.append(test_with_start)
        if val_set:
            if train_set is None:
                train_set = timeserie
            train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=1, shuffle=False)
            datasets.append(val_set)
        datasets.append(train_set)
        # the data set will start with the test set and then all other sets
        datasets.reverse()
        return datasets

    def oxy_architecture_input(self, **kwargs):
        inputs, oxy_out, loss_function = self.architecture(**kwargs)

        return inputs, oxy_out, loss_function

    def compile_model(self, mapper_function, inputs, oxy_out, optimizer, loss, model_loss, **kwargs):
        out_layer = TimeDistributed(Dense(len(self.data_object.target_col), name='bp_out'), name="steps_bp")
        if (not self.parameters['sep_mapper']) and (model_loss is None):
            if mapper_function:
                mapping_layer = mapper_function(oxy_out, **kwargs)
            else:
                mapping_layer = oxy_out
            bp_out = out_layer(mapping_layer)
            # model
            model = keras.Model(inputs=inputs, outputs=[bp_out, oxy_out],
                                name='BP_model')
            if self.m_eager and self.architecture.AR_eager:
                run_eager = True
            else:
                run_eager = False
            model.compile(optimizer=optimizer, loss=loss, metrics=['mae'], run_eagerly=run_eager)
            mapper = None
        else:
            model = keras.Model(inputs=inputs, outputs=oxy_out,
                                name='Oxy_model')
            # if we don't have a model specific loss we apply the normal loss
            if model_loss is None:
                model_loss = loss
            model.compile(optimizer=optimizer, loss=model_loss, metrics=['mae'], run_eagerly=self.architecture.AR_eager)

            # input layer for mapper
            oxy_input = keras.Input(shape=(self.n_out_steps, self.n_in_features),
                                    name='oxy_dxy_in')
            if mapper_function:
                mapping_layer = mapper_function(oxy_input, **kwargs)
            else:
                mapping_layer = oxy_input
            bp_out = out_layer(mapping_layer)
            mapper = keras.Model(inputs=oxy_input, outputs=bp_out,
                                 name='mapper')

            mapper.compile(optimizer=optimizer, loss=loss, metrics=['mae'], run_eagerly=self.m_eager)

        return model, mapper

    def fit(self, train_set: Tuple[np.ndarray, np.ndarray], val_set: Tuple[np.ndarray, np.ndarray], movement_index,
            callbacks: list) -> int:

        early_stopper = EarlyStopping(patience=10, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(factor=0.33, patience=6, min_lr=1e-6, verbose=1)
        nan_terminate = TerminateOnNaN()

        X_train_inputs, y_train_outputs = self._prep_data(train_set, movement_index)
        X_val_inputs, y_val_outputs = self._prep_data(val_set, movement_index)

        if callbacks is None:
            callbacks = [early_stopper, lr_reducer, nan_terminate]
        else:
            callbacks.extend([early_stopper, lr_reducer, nan_terminate])
        if self.mapper:
            history = self.model.fit(x=X_train_inputs, y=y_train_outputs[0],
                                     validation_data=(X_val_inputs, y_val_outputs[0]),
                                     epochs=self.parameters["epochs"],
                                     batch_size=self.parameters["batch_size"],
                                     shuffle=True,
                                     callbacks=callbacks,
                                     verbose=0)
            # get intermediate values from architecture if needed
            self.get_intermediate = self.architecture.get_intermediate_values(self.model)
            gc.collect()

            self.mapper.fit(x=y_train_outputs[0], y=y_train_outputs[1],
                            validation_data=(y_val_outputs[0], y_val_outputs[1]),
                            epochs=self.parameters["epochs"],
                            batch_size=self.parameters["batch_size"],
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=0)
        else:
            history = self.model.fit(x=X_train_inputs, y=y_train_outputs, validation_data=(X_val_inputs, y_val_outputs),
                                     epochs=self.parameters["epochs"],
                                     batch_size=self.parameters["batch_size"],
                                     shuffle=True,
                                     callbacks=callbacks,
                                     verbose=0)

        return len(history.history["loss"])

    def _prep_data(self, dataset, movement_index):
        # X data
        X_data = dataset[0]
        # split up into movement and oxy/dxy input
        cols = list(range(X_data.shape[2]))
        if movement_index:
            movement_columns = np.array(list(movement_index.values())) - len(self.data_object.target_col)
            cols = list(set(cols) - set(movement_columns))

            X_data_mov = X_data[:, :, movement_columns]
            X_data_architecture = X_data[:, :, cols]
            X = [X_data_architecture, X_data_mov]
        else:
            X = X_data[:, :, cols]

        # y data
        y_data = dataset[1]
        # split up into BP and oxy/dxy output
        y_data_mapper = y_data[:, :, :len(self.data_object.target_col)]
        if y_data_mapper.ndim == 2:
            y_data_mapper = np.expand_dims(y_data_mapper, 2)
        y_data_architecture = y_data[:, :, len(self.data_object.target_col):]
        y = [y_data_architecture, y_data_mapper]
        return X, y

    def _make_prediction(self, inputs):
        if self.get_intermediate:
            architecture_pred, std = self.architecture.use_intermediate_values(inputs, self.get_intermediate)
        else:
            architecture_pred = self.model.predict(x=inputs)
            std = None
        return architecture_pred, std

    def predict(self, data, movement_index):
        if isinstance(data, tuple):
            X_test_inputs, _ = self._prep_data(data, movement_index)
        else:
            raise ValueError("Expects tuple of (X,y)")
        # use intermediate values for making the prediction
        if self.mapper:
            architecture_pred, std = self._make_prediction(X_test_inputs)
            mapper_pred = self.mapper.predict(x=architecture_pred)
            prediction = [mapper_pred, architecture_pred]
        else:
            prediction = self.model.predict(x=X_test_inputs)
            std = None
        prediction = np.dstack(prediction)
        return prediction, std

    def predict_future(self, X_train, num_chunks):
        # get last n_past from training
        X_train_data = X_train[:, :, :self.n_in_features]
        X_train_mov = X_train[:, :, self.n_in_features:]
        past_data = X_train_data[-1]
        past_mov = X_train_mov[-1]

        # loop predict
        predictions_features = past_data.copy()
        standard_deviation = []
        predictions_mapper = np.array([])
        for chunk in tqdm(range(num_chunks)):
            input_data = predictions_features[-self.n_in_steps:]
            input_data = np.expand_dims(input_data, axis=0)
            input_mov = np.expand_dims(past_mov[-self.n_in_steps:], axis=0)
            if self.mapper:
                pred_y_architecture, std = self._make_prediction([input_data, input_mov])
                pred_y_mapper = self.mapper.predict(pred_y_architecture)
                if std is not None:
                    standard_deviation.extend(std)
            else:
                pred_y_mapper, pred_y_architecture = self.model.predict([input_data, input_mov])

            predictions_features = np.vstack([predictions_features, pred_y_architecture.squeeze()])
            if chunk == 0:
                predictions_mapper = pred_y_mapper[0]
            else:
                predictions_mapper = np.vstack([predictions_mapper, pred_y_mapper[0]])

            # future movement
            past_mov = get_movement(past_mov, pred_y_architecture)

        # merge architect and mapper output
        # remove starting data
        predictions_features = predictions_features[self.n_in_steps:]
        predictions_list = np.hstack([predictions_mapper, predictions_features])
        if standard_deviation:
            standard_deviation = np.array(standard_deviation)
        else:
            standard_deviation = None
        return predictions_list, standard_deviation

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, arg=None):
        self.parameters = self.get_keras_parameters(arg)
        if arg is None:
            self.logger.info("setting default variables")
            self._set_default_parameters()
        elif isinstance(arg, dict):
            self.logger.info("updating default variables with parameters")
            self._set_default_parameters()
            parameters = self.parameters
            parameters.update(arg)
            try:
                parameters.pop("Outlier detection")
            except KeyError:
                pass
            self.parameters = parameters
        else:
            self.logger.info("Setting with optuna")
            self._set_optuna_parameters(arg)

    def _set_optuna_parameters(self, trial: Trial):
        raise NotImplementedError("This is model specific, needs to be called from model class")

    def _set_default_parameters(self):
        raise NotImplementedError("This is model specific, needs to be called from model class")

    def get_keras_parameters(self, args: Optional[Union[dict, Trial]] = None):
        epochs = Parameters.iterations.value
        batch_size = Parameters.batch_size.value
        if isinstance(args, Trial):
            opti = args.suggest_categorical("optimizer", ["SGD", "RMSprop", "Adam"])
            if opti == "SGD":
                opti = SGD
            elif opti == "RMSprop":
                opti = RMSprop
            else:
                opti = Adam

            architecture_name = args.suggest_categorical("architecture_name", [arch.value for arch in Architectures])

            self.architecture = ArchitectureCreator.create_architecture(architecture_name,
                                                                        n_in_steps=self.n_in_steps,
                                                                        n_in_features=self.n_in_features,
                                                                        n_out_steps=self.n_out_steps,
                                                                        data_object=self.data_object)
            sep_mapper = self.architecture.sep_mapper
            if sep_mapper is None:
                sep_mapper = args.suggest_categorical("sep_mapper", [True, False])
            else:
                args.set_user_attr("sep_mapper", sep_mapper)
            architecture_params = self.architecture.get_trial_parameters(args)

            loss = "mae"
            args.set_user_attr("loss", loss)

            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "activation": 'relu',
                "sep_mapper": sep_mapper,
                "optimizer": opti(),
                "loss": loss
            }
            architecture_params.update({"architecture_name": architecture_name})
            keras_parameters.update(architecture_params)
        elif isinstance(args, dict):
            architecture_name = args["architecture_name"]
            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "activation": 'relu',
                "sep_mapper": args["sep_mapper"],
                "optimizer": args["optimizer"],
                "loss": args["loss"]
            }
            self.architecture = ArchitectureCreator.create_architecture(architecture_name,
                                                                        n_in_steps=self.n_in_steps,
                                                                        n_in_features=self.n_in_features,
                                                                        n_out_steps=self.n_out_steps,
                                                                        data_object=self.data_object)
            architecture_params = self.architecture.get_parameters()
            architecture_params.update({"architecture_name": architecture_name})
            keras_parameters.update(architecture_params)
        else:
            architecture_name = "basic"
            keras_parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "activation": 'relu',
                "sep_mapper": True,
                "optimizer": "adam",
                "loss": "mse"
            }
            self.architecture = ArchitectureCreator.create_architecture(architecture_name,
                                                                        n_in_steps=self.n_in_steps,
                                                                        n_in_features=self.n_in_features,
                                                                        n_out_steps=self.n_out_steps,
                                                                        data_object=self.data_object)
            architecture_params = self.architecture.get_parameters()
            architecture_params.update({"architecture_name": architecture_name})
            keras_parameters.update(architecture_params)
        self.logger.debug(architecture_name)
        return keras_parameters

    def save_model(self):
        self.model.save_weights("model.h5")
        if self.mapper:
            self.mapper.save_weights("mapper.h5")

    def load_model(self, folder_name):
        # contains bug
        file_name = os.path.join(folder_name, "model.h5")
        self.model.load_weights(file_name)
        if self.mapper:
            file_name = os.path.join(folder_name, "mapper.h5")
            self.mapper.load_weights(file_name)
        # get intermediate values from architecture if needed
        self.get_intermediate = self.architecture.get_intermediate_values(self.model)


class Base(KerasModel):
    def __init__(self, data_object: DataObject, n_in_steps, n_out_steps: int, n_in_features: int,
                 n_mov_features: int, parameters: Optional[dict], plot_layers: bool,
                 gpu):
        get_logger().setLevel('ERROR')
        super().__init__()
        self.logger.debug(f"GPU: {gpu}")

        self.n_in_steps = n_in_steps
        self.n_in_features = n_in_features
        self.n_mov_features = n_mov_features
        self.n_out_steps = n_out_steps
        self.data_object = data_object

        # fill parameters
        # todo: get_keras_parameters called twice and not all parameters are copied from the optimization run
        self.set_parameters(parameters)

        self.model, self.mapper = self._architecture(**self.parameters)
        if plot_layers:
            plot_model(self.model, show_shapes=True, to_file="model.png")
            if self.mapper:
                plot_model(self.mapper, show_shapes=True, to_file="mapper.png")

    def _architecture(self, optimizer, loss,
                      **kwargs):
        # Creating the layers
        inputs, oxy_out, model_loss = self.oxy_architecture_input(**kwargs)

        model, mapper = self.compile_model(self._get_model(), inputs, oxy_out, optimizer, loss, model_loss, **kwargs)
        return model, mapper

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError("Must override method")

    def _set_optuna_parameters(self, trial: Trial):
        pass

    def _set_default_parameters(self):
        pass
