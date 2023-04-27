import tensorflow as tf
from tensorflow import keras

class Meso4():
    def __init__(self, output_directory, input_shape, nb_classes, verbose):
        self.callbacks = None
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.model.summary()
        self.verbose = verbose
        self.nb_classes = nb_classes

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = keras.layers.Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = keras.layers.BatchNormalization()(x2)
        x2 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = keras.layers.Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = keras.layers.BatchNormalization()(x3)
        x3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = keras.layers.Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = keras.layers.BatchNormalization()(x4)
        x4 = keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = keras.layers.Flatten()(x4)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Dense(16)(y)
        y = keras.layers.LeakyReLU(alpha=0.1)(y)
        y = keras.layers.Dropout(0.5)(y)
        if nb_classes == 1:
            y = keras.layers.Dense(nb_classes, activation='sigmoid')(y)
        else:
            y = keras.layers.Dense(nb_classes, activation='softmax')(y)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        self.callbacks = [model_checkpoint]
        model = keras.models.Model(inputs=input_layer, outputs=y)

        return model

    def save(self):
        self.model.save(self.output_directory + "model_%s" % self.nb_classes)

class MesoInception4():
    def __init__(self, output_directory, input_shape, nb_classes, verbose):
        self.callbacks = None
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.model.summary()
        self.verbose = verbose
        self.nb_classes = nb_classes

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = keras.layers.Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = keras.layers.Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = keras.layers.Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = keras.layers.Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = keras.layers.Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = keras.layers.Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = keras.layers.Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = keras.layers.Concatenate(axis=-1)([x1, x2, x3, x4])

            return y
        return func

    def build_model(self, input_shape, nb_classes):
        x = keras.layers.Input(input_shape)

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = keras.layers.BatchNormalization()(x2)
        x2 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = keras.layers.BatchNormalization()(x3)
        x3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = keras.layers.BatchNormalization()(x4)
        x4 = keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = keras.layers.Flatten()(x4)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Dense(16)(y)
        y = keras.layers.LeakyReLU(alpha=0.1)(y)
        y = keras.layers.Dropout(0.5)(y)
        if nb_classes == 1:
            y = keras.layers.Dense(1, activation='sigmoid')(y)
        else:
            y = keras.layers.Dense(2, activation='softmax')(y)
        model = keras.models.Model(inputs=x, outputs=y)
        return model

    def save(self):
        self.model.save(self.output_directory + "model_%s" % self.nb_classes)