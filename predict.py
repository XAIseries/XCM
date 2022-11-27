import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils.data_loading import import_data
from utils.explainability import get_heatmap
from utils.helpers import create_logger, save_experiment

from models.mtex_cnn import mtex_cnn
from models.xcm import xcm
from models.xcm_seq import xcm_seq
from tensorflow import keras



if __name__ == "__main__":

    # Load configuration
    parser = argparse.ArgumentParser(description="XCM")
    parser.add_argument(
        "-c", "--config", default="configuration/config.yml", help="Configuration File"
    )
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configuration = yaml.safe_load(config_file)

    if configuration["model_name"] in ["XCM", "XCM-Seq"]:
        window_size = configuration["window_size"]
    else:
        window_size = 0
    model_dict = {"XCM": xcm, "XCM-Seq": xcm_seq, "MTEX-CNN": mtex_cnn}

    # Create experiment folder
    xp_dir = (
        "./results/"
        + str(configuration["dataset"])
        + "/"
        + str(configuration["model_name"])
        + "/XP_"
        + str(configuration["experiment_run"])
        + "/"
    )
    save_experiment(xp_dir, args.config)
    log, logclose = create_logger(log_filename=os.path.join(xp_dir, "experiment.log"))
    log("Model: " + configuration["model_name"])

    # Load dataset
    (
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_nonencoded,
        y_test_nonencoded,
    ) = import_data(configuration["dataset"], log)
    print("X_train.shape: ", X_train.shape)
    print("X_test.shape: ", X_test.shape)

    model = keras.models.load_model(xp_dir + "/model.h5")
    # Example of a heatmap from Grad-CAM for the first MTS of the test set
    get_heatmap(
        configuration,
        xp_dir,
        model,
        X_train,
        X_test,
        y_train_nonencoded,
        y_test_nonencoded,
    )

    logclose()
