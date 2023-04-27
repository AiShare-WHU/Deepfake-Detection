import tensorflow as tf
from tensorflow import keras


class MobileNet():
    def __init__(self, output_directory, input_shape, nb_classes, verbose):
        self.callbacks = None
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.model.summary()
        self.verbose = verbose
        self.nb_classes = nb_classes

    def build_model(self, input_shape, nb_classes):
        MobileNetV3Small = keras.applications.MobileNetV3Small(input_shape=input_shape, weights=None, classes=nb_classes)
        if nb_classes == 1:
            output = keras.activations.sigmoid(MobileNetV3Small.output)
            model = keras.models.Model(inputs=MobileNetV3Small.input, outputs=output)
        else:
            # output = keras.activations.softmax(Xception.output)
            model = keras.models.Model(inputs=MobileNetV3Small.input, outputs=MobileNetV3Small.output)
        return model

    def save(self):
        self.model.save(self.output_directory + "model_%s" % self.nb_classes)
