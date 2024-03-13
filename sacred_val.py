from sacred import Experiment
from train import *
from test import *
from tensorflow import keras

ex = Experiment("val")


@ex.config
def my_config():
    model_name = "Meso4"
    batch_size = 64
    epochs = 50
    data_shape = (256, 256, 3)
    nb_class = 2
    learning_rate = 0.001
    data_set = "Deepfakes"
    save_path = "./saved_data/"
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
    train_data_path = "./data/%s/train_images/" % data_set
    val_data_path = "./data/%s/val_images/" % data_set
    save_path = save_path + data_set + "/"
    if data_set == "NeuralTextures" and model_name == "EfficientNet":
        model = keras.models.load_model(
            "./saved_data/" + data_set + "/" + model_name + "/"
        )
    else:
        model = keras.models.load_model(
            "./saved_data/" + data_set + "/" + model_name + "/model_2.hdf5"
        )
    if data_set == "DFDC":
        ff, fr, rf, rr = test_on_subset(
            model, "./data/", data_set, "train_images", data_shape[0:2]
        )
    else:
        ff, fr, rf, rr = test_on_subset(
            model, "./data/", data_set, "val_images", data_shape[0:2]
        )
    return (
        "Model %s on val_data %s : Accuracy = %s, precision = %s, recall = %s, F~F=%s, F~R=%s, R~F=%s, R~R=%s"
        % (
            model_name,
            data_set,
            str((ff + rr) / (ff + fr + rf + rr)),
            str(ff / (ff + rf + 1)),
            str(ff / (ff + fr + 1)),
            str(ff),
            str(fr),
            str(rf),
            str(rr),
        )
    )
