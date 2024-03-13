from sacred import Experiment
from train import *
from test import *

ex = Experiment("train")


@ex.config
def my_config():
    model_name = "Meso4"
    model_size = {
        "Meso4": (256, 256, 3),
        "MesoInception4": (256, 256, 3),
        "Xception": (299, 299, 3),
        "MobileNet": (224, 224, 3),
        "EfficientNet": (224, 224, 3),
    }
    batch_size = 32
    epochs = 100
    data_shape = model_size[model_name]
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
    if onehot:
        label_mode = "categorical"
    else:
        label_mode = "binary"
    train_data = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        batch_size=batch_size,
        image_size=data_shape[0:2],
        shuffle=True,
        seed=42,
        validation_split=None,
        subset=None,
    )
    if data_set == "DFDC" or data_set == "FMFCC":
        val_data = None
    else:
        val_data = tf.keras.utils.image_dataset_from_directory(
            val_data_path,
            labels="inferred",
            label_mode=label_mode,
            class_names=None,
            batch_size=batch_size,
            image_size=data_shape[0:2],
            shuffle=False,
            seed=42,
            validation_split=None,
            subset=None,
        )
    model = create_model(model_name, data_shape, nb_class, save_path)
    train_model(model.model, train_data, val_data, learning_rate, epochs, onehot)
    model.save()
    ff, fr, rf, rr, video_results = test_on_subset(
        model.model, "./data/", data_set, "train_images", data_shape[0:2]
    )
    return (
        "Accuracy = %s, precision = %s, recall = %s, F~F=%s, F~R=%s, R~F=%s, R~R=%s"
        % (
            str((ff + rr) / (ff + fr + rf + rr)),
            str(ff / (ff + rf)),
            str(ff / (ff + fr)),
            str(ff),
            str(fr),
            str(rf),
            str(rr),
        )
    )
