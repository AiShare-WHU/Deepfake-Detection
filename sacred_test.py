import json
import os.path

import keras.models
from sacred import Experiment
from sacred.observers import SlackObserver
from train import *
from test import *

ex = Experiment("test")


@ex.config
def my_config():
    model_name = "Meso4"
    batch_size = 64
    epochs = 50
    data_shape = (256, 256, 3)
    nb_class = 2
    learning_rate = 0.001
    data_set = "Face2Face"
    save_path = "./result/"
    onehot = True


@ex.automain
def my_main(
    model_name,
    batch_size,
    epochs,
    data_shape,
    nb_class,
    learning_rate,
    data_set,
    save_path,
    onehot,
):
    if model_name == "EfficientNet" and data_set == "NeuralTextures":
        model = keras.models.load_model(
            "./saved_data/" + data_set + "/" + model_name + "/"
        )
    else:
        model = keras.models.load_model(
            "./saved_data/" + data_set + "/" + model_name + "/model_2.hdf5"
        )
    subsets = [
        "test_images",
        "fl2_test_images",
        "rn2_test_images",
        "ei2_test_images",
        "ce2_test_images",
    ]
    for subset in subsets:
        if subset == "test_images":
            ff, fr, rf, rr, result = test_on_subset(
                model, "./data/", data_set, subset, data_shape[0:2]
            )
            if not os.path.exists(
                save_path + model_name + "/" + data_set + "/" + subset + "/"
            ):
                os.makedirs(
                    save_path + model_name + "/" + data_set + "/" + subset + "/"
                )
            with open(
                save_path + model_name + "/" + data_set + "/" + subset + "/result.json",
                "w",
            ) as f:
                json.dump(result, f)
            print(
                "model %s on dataset %s on subset %s: Accuracy = %s, precision = %s, recall = %s, F~F=%s, F~R=%s, "
                "R~F=%s, R~R=%s"
                % (
                    model_name,
                    data_set,
                    subset,
                    str((ff + rr) / (ff + fr + rf + rr)),
                    str(ff / (ff + rf + 1)),
                    str(ff / (ff + fr + 1)),
                    str(ff),
                    str(fr),
                    str(rf),
                    str(rr),
                )
            )
        else:
            for rate in os.listdir("./data/" + data_set + "/" + subset):
                ff, fr, rf, rr, result = test_on_subset(
                    model, "./data/", data_set, subset + "/" + rate, data_shape[0:2]
                )
                if not os.path.exists(
                    save_path
                    + model_name
                    + "/"
                    + data_set
                    + "/"
                    + subset
                    + "/"
                    + rate
                    + "/"
                ):
                    os.makedirs(
                        save_path
                        + model_name
                        + "/"
                        + data_set
                        + "/"
                        + subset
                        + "/"
                        + rate
                        + "/"
                    )
                with open(
                    save_path
                    + model_name
                    + "/"
                    + data_set
                    + "/"
                    + subset
                    + "/"
                    + rate
                    + "/result.json",
                    "w",
                ) as f:
                    json.dump(result, f)
                print(
                    "model %s on dataset %s on subset %s with rate %s: Accuracy = %s, precision = %s, recall = %s, F~F=%s, F~R=%s, "
                    "R~F=%s, R~R=%s"
                    % (
                        model_name,
                        data_set,
                        subset,
                        rate,
                        str((ff + rr) / (ff + fr + rf + rr)),
                        str(ff / (ff + rf + 1)),
                        str(ff / (ff + fr + 1)),
                        str(ff),
                        str(fr),
                        str(rf),
                        str(rr),
                    )
                )
