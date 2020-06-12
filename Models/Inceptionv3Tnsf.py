import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization


def model():
    # define the basemodel
    base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(440, 440, 3))
    for layers in base_model.layers:
        layers.trainable = False

    # print(base_model.get_layer('conv2d_80').output_shape)

    x = Conv2D(filters=128, activation='relu', kernel_size=1)(base_model.get_layer('conv2d_80').output)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, activation='relu',
               kernel_size=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, activation='relu',
               kernel_size=1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=512, activation='relu',
               kernel_size=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=1024, activation='relu',
               kernel_size=3)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=1024, activation='relu',
               kernel_size=3, strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=2048, activation='relu',
               kernel_size=1)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dropout(0.4)(x)

    x = Dense(512, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.4)(x)

    x = Dense(256, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.4)(x)
    x = Dense(128, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.4)(x)

    x = Dense(8, activation=tf.nn.softmax)(x)

    MoDel = tf.keras.Model(base_model.input, x)

    MoDel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return MoDel
