# Author: Enrico Schmitz (s1047521)
# Master Thesis Data science
# Project: Applications of Deep Learning on Orthostatic Hypotension detection
# Assignment of: Donders Institute for Brain, Cognition and Behaviour
# Script: Rewrite the MLflow yaml to correspond to the current path

# Imports
import getopt
import glob

import sys
from pathlib import Path

# Variables
MLFLOW_NAME = 'klop'
DIR_NAME = "C:/Users/enric/Desktop/Final_results/"
OLD_PATH = "/home/eschmitz/"
FILE_TYPE = "*.yaml"


def main(argv):
    # parse arguments
    args, usage = parse_arguments(argv)
    if "dir_name" in args:
        dir_name = args["dir_name"]
    else:
        dir_name = DIR_NAME
    if "mlflow_name" in args:
        mlflow_name = args["mlflow_name"]
    else:
        mlflow_name = MLFLOW_NAME
    if "old_path" in args:
        old_path = args["old_str"]
    else:
        old_path = OLD_PATH

    arguments = {
        "dir_name": dir_name,
        "mlflow_name": mlflow_name,
        "old_path": old_path
    }
    # assert we have a value for all parameters
    assert (dir_name is not None), not_specified("dir_name", usage, arguments)
    assert (mlflow_name is not None), not_specified("mlflow_name", usage, arguments)
    assert (old_path is not None), not_specified("old_path", usage, arguments)

    # get all files in directories, recursive
    pathname = f"{dir_name}{mlflow_name}/**/{FILE_TYPE}"
    files = glob.glob(pathname, recursive=True)
    for path in files:
        path = Path(path)
        # Read in the file
        with open(path, "r") as file:
            file_data = file.read()

        # Replace the target string
        file_data = file_data.replace(old_path, dir_name)

        # Write the file out again
        with open(path, "w") as file:
            file.write(file_data)


def not_specified(variable_name, usage, arguments):
    return f"{variable_name} is not specified; \n usage:{usage} \n got:{arguments}"


def parse_arguments(argv):
    usage = "rewrite_mlflow_yaml.py -i <dir_name> -m <mlflow_name> -o <old_path>"
    try:
        opts, args = getopt.getopt(argv,
                                   "himo",
                                   [
                                       "dir_name=",
                                       "mlflow_name="
                                       "old_path="
                                   ]
                                   )
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    args_dict = {}
    if opts:
        for opt, arg in opts:
            if opt == "-h":
                print(usage)
                sys.exit()
            elif opt in ("-i", "--dir_name"):
                args_dict["dir_name"] = arg
            elif opt in ("-m", "--mlflow_name"):
                args_dict["mlflow_name"] = arg
            elif opt in ("-o", "--old_path"):
                args_dict["old_path"] = arg
    return args_dict, usage


if __name__ == "__main__":
    main(sys.argv[1:])
