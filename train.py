import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from model import MesoNet, Xception, EfficientNet, MobileNet


def create_model(model_name, data_shape, nb_class, save_path):
    save_model_path = save_path + model_name + "/"
    if model_name == "Meso4":
        model = MesoNet.Meso4(save_model_path, data_shape, nb_class, 1)
    elif model_name == "MesoInception4":
        model = MesoNet.MesoInception4(save_model_path, data_shape, nb_class, 1)
    elif model_name == "Xception":
        model = Xception.Xception(save_model_path, data_shape, nb_class, 1)
    elif model_name == "EfficientNet":
        model = EfficientNet.EfficientNet(save_model_path, data_shape, nb_class, 1)
    elif model_name == "MobileNet":
        model = MobileNet.MobileNet(save_model_path, data_shape, nb_class, 1)
    else:
        raise NameError("model %s not found")
    return model


def train_model(model, dataset, val_dataset, learning_rate, epochs, onehot):
    if onehot:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=keras.metrics.Accuracy())
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=keras.metrics.Accuracy())
    if val_dataset == None:
        model.fit(dataset, epochs=epochs)
    else:
        model.fit(dataset, validation_data=val_dataset, epochs=epochs)


if __name__ == "__main__":
    model_size = {"Meso4": (256, 256, 3), "MesoInception4": (256, 256, 3), "Xception": (299, 299, 3),
                  "MobileNet": (224, 224, 3), "EfficientNet": (224, 224, 3)}
    model_name = "Meso4"
    batch_size = 64
    epochs = 50
    data_shape = model_size[model_name]
    nb_class = 1
    learning_rate = 0.001
    data_path = "./data/Deepfakes/train_images/"
    save_path = "./saved_data/"
    onehot = True
    if onehot:
        label_mode = "categorical"
    else:
        label_mode = "binary"
    data = tf.keras.utils.image_dataset_from_directory(data_path, labels='inferred', label_mode=label_mode,
                                                       class_names=None, batch_size=batch_size,
                                                       image_size=data_shape[0:2], shuffle=False, seed=None,
                                                       validation_split=None, subset=None, )
    model = create_model(model_name, data_shape, nb_class, save_path)
    train_model(model.model, data, learning_rate, epochs, onehot)
    model.save()
    print("Done")
