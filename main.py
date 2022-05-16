# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Main script to call functionalities

# Imports
import getopt
import logging
import os
import sys
import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Detector.Task.Optimize_model import optimize_model
from Detector.Task.Preprocessing import preprocessing
from Detector.Task.Train_model import train_model
from Detector.Utility.Task.preprocessing.PreprocessingFunctions import create_info_object
from Detector.enums import MLModelType

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.ERROR)
# variables
dataset_name = "klop"
save_file = False
plot_BP = False
plot_features = False
OP = True
FI = True
LoopModels = False


def main(argv):
    # get arguments
    Optimize, Fit, info_dict = parse_arguments(argv)

    # create an object containing info from json
    info_object = create_info_object(dataset_name, info_dict)

    # preprocess the data
    data_object, X, info_dataset, parameters, full_curve = preprocessing(info_object)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(full_curve))

    data_object.scaler = MinMaxScaler(feature_range=(-1, 1))

    # if LoopModels is true we perform it on all models
    if LoopModels:
        models = MLModelType
    # if False we only want de model specified in the json
    else:
        models = [info_object.model]

    # Loop over models
    for model in models:
        if LoopModels:
            model = model.value
        info_object.model = model

        # set model type correct
        if info_object.model in ["deepar", "nbeats"]:
            info_object.parameter_model = False
        else:
            info_object.parameter_model = True

        # optimize a model
        if Optimize:
            optimize_model(X, info_dataset, parameters, full_curve, data_object, info_object)
        if Fit or Optimize:
            # fit a model
            train_model(X, info_dataset, parameters, full_curve, data_object, info_object)
        # predict future
        # predict_future(df, n_in_steps, n_out_steps, n_features, info_object, data_object, tags, scaler)


def parse_arguments(argv):
    usage = 'main.py -i <input_steps> -o <output_steps> -d <dataset_name> ' \
            '-m <model> -a <outlier_detection> ' \
            '-s <resample>  -w <rolling_window> ' \
            '-l <lagging>  -f <file_loc>'
    Optimize = None
    Fit = None
    try:
        opts, args = getopt.getopt(argv,
                                   "hiodmaswlfpt",
                                   [
                                       "input_steps=",
                                       "output_steps=",
                                       "dataset=",
                                       "model=",
                                       "outlier_detection=",
                                       "resample=",
                                       "rolling_window=",
                                       "lagging=",
                                       "file_loc=",
                                       "optimize=",
                                       "fit="
                                   ])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    info_dict = {}
    if opts:
        for opt, arg in opts:
            if opt == '-h':
                print(usage)
                sys.exit()
            elif opt in ("-i", "--input_steps"):
                info_dict["input_steps"] = arg
            elif opt in ("-o", "--output_steps"):
                info_dict["output_steps"] = arg
            elif opt in ("-d", "--dataset"):
                info_dict["dataset"] = arg
            elif opt in ("-m", "--model"):
                info_dict["model"] = arg
            elif opt in ("-a", "--outlier_detection"):
                info_dict["outlier_algo"] = arg
            elif opt in ("-s", "--resample"):
                info_dict["resample"] = arg
            elif opt in ("-w", "--rolling_window"):
                info_dict["rolling_window"] = arg
            elif opt in ("-l", "--lagging"):
                info_dict["lagging"] = arg
            elif opt in ("-f", "--file_loc"):
                info_dict["file_loc"] = arg
            elif opt in ("-p", "--optimize"):
                Optimize = parse_bool(opt, arg)
            elif opt in ("-t", "--fit"):
                Fit = parse_bool(opt, arg)
    if Optimize is None:
        Optimize = OP
    if Fit is None:
        Fit = FI
    return Optimize, Fit, info_dict


def parse_bool(opt, arg):
    if arg.lower() == "true":
        arg = True
    elif arg.lower() == "false":
        arg = False
    else:
        raise ValueError(f"Unknown {opt} value, True or False only, given={arg}")
    return arg


if __name__ == '__main__':
    os.chdir(os.getcwd())
    main(sys.argv[1:])
