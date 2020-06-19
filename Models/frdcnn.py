import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import Model

tf.config.experimental_run_functions_eagerly(True)


def _fire_block(input_tensor, s1, e1, e3):
    # _, H, W, _ = input_tensor.shape

    # squeeze layer
    # 1x1 kernel with s1 channels
    output = Conv2D(filters=s1, kernel_size=(1, 1), padding='same', activation='relu',
                    kernel_initializer=tf.keras.initializers.glorot_uniform())(input_tensor)

    # assert output.shape == (H, W, s1), "shape of squeeze layer isn't equal"

    # Start Expand layer
    # output will be fed into one 1x1 conv and 3x3 conv
    # 1x1 expansion
    output1 = Conv2D(filters=e1, kernel_size=(1, 1), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.glorot_uniform())(output)

    # 3x3 expansion
    output3 = Conv2D(filters=e3, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.glorot_uniform())(output)

    # Concatenate the parallel expansion layers
    output = tf.concat([output1, output3], axis=-1)

    # assert output.shape == (H, W, (e1 + e3)), "shape of Expansion layer isn't equal"

    return output


def _block(input_tensor, s1, e1, e3, name=None):
    """
    Contains fire module 2 and 3
    :return: Addition of fire2 + fire3 via skip connection
    """
    fire2 = _fire_block(input_tensor, s1, e1, e3)
    fire3 = _fire_block(fire2, s1, e1, e3)

    # add fire two and fire 3
    output = tf.add(fire2, fire3, name=name)

    return output


def _block_three(input_tensor):
    """
    Starts with One max pooling, contains fire6-10
    :param input_tensor: fire5
    :return: fire10 output
    """
    input_tensor = MaxPooling2D(pool_size=(3, 3), strides=2)(input_tensor)
    _S69, _E169, _E369 = 288, 192, 192
    _S10, _E110, _E310 = 384, 256, 256

    fire6 = _fire_block(input_tensor, s1=_S69, e1=_E169, e3=_E369)
    fire7 = _fire_block(fire6, s1=_S69, e1=_E169, e3=_E369)
    concat67 = tf.concat([fire6, fire7], axis=-1)

    fire8 = _fire_block(concat67, s1=_S69, e1=_E169, e3=_E369)
    concat678 = tf.concat([fire6, fire7, fire8], axis=-1)

    fire9 = _fire_block(concat678, s1=_S69, e1=_E169, e3=_E369)
    concat6789 = tf.concat([fire6, fire7, fire8, fire9], axis=-1)

    fire10 = _fire_block(concat6789, s1=_S10, e1=_E110, e3=_E310)

    return fire10
    pass


def _block_four(input_tensor):
    """
    Starts with one Max Pool, contains fire11-15
    :param input_tensor: fire10
    :return: fire15
    """
    input_tensor = MaxPooling2D(pool_size=(3, 3), strides=2)(input_tensor)

    # default block parameters
    _S14, _E114, _E314 = 288, 192, 192
    _S115, _E115, _E315 = 384, 256, 256

    fire11 = _fire_block(input_tensor, s1=_S14, e1=_E114, e3=_E314)
    fire12 = _fire_block(fire11, s1=_S14, e1=_E114, e3=_E314)
    concat12 = tf.concat([fire11, fire12], axis=-1)

    fire13 = _fire_block(concat12, s1=_S14, e1=_E114, e3=_E314)
    concat123 = tf.concat([fire11, fire12, fire13], axis=-1)

    fire14 = _fire_block(concat123, s1=_S14, e1=_E114, e3=_E314)
    concat1234 = tf.concat([fire11, fire12, fire13, fire14], axis=-1)

    fire15 = _fire_block(concat1234, s1=_S115, e1=_E115, e3=_E315)

    return fire15
    pass


class frdcnn(Model):

    def __init__(self, num_classes):
        super(frdcnn, self).__init__()

        self.conv1 = Conv2D(filters=96, kernel_size=7, strides=2, padding='valid', activation='relu',
                            kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')
        self.convfinal = Conv2D(kernel_size=(3, 3), filters=9 * num_classes, padding='same', activation='relu',
                                kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.flatten = Flatten()
        self.dense1 = Dense(units=2048, activation='relu')
        self.dense2 = Dense(units=512, activation='relu')
        self.classify = Dense(units=8, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpool1(x)

        # block one
        x = _block(x, s1=96, e1=64, e3=64)

        # block two
        x = _block(x, s1=192, e1=128, e3=128)

        # block three
        x = _block_three(x)

        # block four
        x = _block_four(x)

        # final conv
        x = self.convfinal(x)

        # fc layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.classify(x)

        return x
