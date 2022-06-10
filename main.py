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

from sklearn.preprocessing import MinMaxScaler

from Detector.Task.Optimize_model import optimize_model
from Detector.Task.Preprocessing import preprocessing
from Detector.Task.Train_model import train_model
from Detector.Utility.Task.model_functions import filter_out_test_subjects
from Detector.Utility.Task.preprocessing.PreprocessingFunctions import create_info_object
from Detector.enums import MLModelType

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
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
        print(info_object.model)
        if info_object.model == "timemlp":
            error = "The baseline_tuple[0]/rest_length and standing_length/future_seconds need to be the same"
            assert full_curve.shape[1] == X.shape[1], error
            info_object.parameter_model = False
        else:
            info_object.parameter_model = True
        # filter out 1 of each group for testing
        fit_indexes, test_indexes = filter_out_test_subjects(info_dataset)

        # optimize a model
        if Optimize and info_object.model != "linearregression":
            optimize_model(x=X, parameters_values=parameters, full_curve=full_curve, info_dataset=info_dataset,
                           data_object=data_object, info_object=info_object, fit_indexes=fit_indexes)
        if Fit or Optimize:
            # fit a model
            train_model(x=X, info_dataset=info_dataset, parameters_values=parameters, full_curve=full_curve,
                        data_object=data_object, info_object=info_object, fit_indexes=fit_indexes,
                        test_indexes=test_indexes)


def parse_arguments(argv):
    usage = "main.py -d <dataset_name> -m <model> " \
            "-o <optimize> -f <fit>" \
            "-s <smooth>   -l <file_loc>"
    Optimize = None
    Fit = None
    try:
        opts, args = getopt.getopt(argv,
                                   "hdmofsnl",
                                   [
                                       "dataset=",
                                       "model=",
                                       "smooth=",
                                       "nirs_input=",
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
            if opt == "-h":
                print(usage)
                sys.exit()
            elif opt in ("-d", "--dataset"):
                info_dict["dataset"] = arg
            elif opt in ("-m", "--model"):
                info_dict["model"] = arg
            elif opt in ("-s", "--smooth"):
                arg = parse_bool(opt, arg)
                info_dict["smooth"] = arg
            elif opt in ("-n", "--nirs_input"):
                arg = parse_bool(opt, arg)
                info_dict["nirs_input"] = arg
            elif opt in ("-l", "--file_loc"):
                info_dict["file_loc"] = arg
            elif opt in ("-o", "--optimize"):
                Optimize = parse_bool(opt, arg)
            elif opt in ("-f", "--fit"):
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


if __name__ == "__main__":
    os.chdir(os.getcwd())
    main(sys.argv[1:])
