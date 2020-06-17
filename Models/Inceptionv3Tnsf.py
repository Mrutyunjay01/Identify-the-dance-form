import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D


def model():
    # define the basemodel
    basemodel = tf.keras.applications.InceptionV3(include_top=False,
                                                  input_shape=(440, 440, 3),
                                                  weights='imagenet')

    # add global pooling
    new_output = GlobalAveragePooling2D()(basemodel.output)

    # add dense layer for classification
    new_output = Dense(units=8, activation='softmax')(new_output)

    MoDel = tf.keras.Model(basemodel.inputs, new_output)

    # set all layers trainable by default
    for layer in MoDel.layers:
        layer.trainable = True

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9

    # fix deep layers
    for layer in MoDel.layers[:-50]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    MoDel.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adamax(lr=1e-2),
        metrics=['accuracy']
    )

    return MoDel
