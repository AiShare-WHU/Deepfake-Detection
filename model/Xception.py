import tensorflow as tf
from tensorflow import keras

class Xception():
    def __init__(self, output_directory, input_shape, nb_classes, verbose):
        self.callbacks = None
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.model.summary()
        self.verbose = verbose
        self.nb_classes = nb_classes

    def build_model(self, input_shape, nb_classes):
        Xception = keras.applications.Xception(input_shape=input_shape, weights=None, classes=nb_classes)
        if nb_classes == 1:
            output = keras.activations.sigmoid(Xception.output)
            model = keras.models.Model(inputs=Xception.input, outputs=output)
        else:
            # output = keras.activations.softmax(Xception.output)
            model = keras.models.Model(inputs=Xception.input, outputs=Xception.output)
        return model

    def save(self):
        self.model.save(self.output_directory + "model_%s" % self.nb_classes)
