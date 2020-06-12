import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout


def model():
    # define the basemodel
    base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(440, 440, 3))

    x = Flatten()(base_model.output)

    x = Dropout(0.3)(x)

    x = Dense(512, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.25)(x)

    x = Dense(256, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.25)(x)
    x = Dense(128, activation=tf.nn.leaky_relu)(x)

    x = Dropout(0.25)(x)

    x = Dense(8, activation=tf.nn.softmax)(x)

    MoDel = tf.keras.Model(base_model.input, x)

    MoDel.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.01),
                  metrics=['accuracy'])

    return MoDel
