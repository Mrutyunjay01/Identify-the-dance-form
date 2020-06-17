import tensorflow as tf

_LEAKY_RELU = 0.01
_EPSILON = 1e-05
_DECAY = 0.9


def upsample(inputs, out_shape, data_format):
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        height = out_shape[3]
        width = out_shape[2]
    else:
        height = out_shape[2]
        width = out_shape[1]

    inputs = tf.image.resize(inputs, (height, width), method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def fixedPadding(inputs, kernel_size, data_format):
    """
    Pads the input along the spatial dimensions independently of input size
    :param inputs: Tensor input
    :param kernel_size: kernel to be used in the Conv2D or MaxPool2D
    :param data_format: channels_last or channels_first
    :return: A tensor with the same format as input
    """

    pad_total = kernel_size - 1
    pad_beginnig = pad_total // 2
    pad_end = pad_total - pad_beginnig

    if data_format == 'channels_first':
        pad_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                     [pad_beginnig, pad_end],
                                     [pad_beginnig, pad_end]])

    else:
        pad_inputs = tf.pad(inputs, [[0, 0],
                                     [pad_beginnig, pad_end],
                                     [pad_beginnig, pad_end],
                                     [0, 0]])

    return pad_inputs


def conv2D_fiexed_padding(inputs, filters, kernel_size, strides=1):
    """ strided convolution with explicit padding """
    if strides > 1:
        inputs = fixedPadding(inputs, kernel_size)

    return tf.keras.layers.Conv2D(
        inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same' if strides == 1 else 'valid',
        use_bias=False
    )
    pass


def batchNorm(inputs):
    """
    Performs bn using standard set of parameters
    :param inputs: input images
    :return: bn applied
    """
    return tf.keras.layers.BatchNormalization(inputs=inputs,
                                              momentum=_DECAY,
                                              epsilon=_EPSILON,
                                              scale=True)


def darknet53_residualBlock(inputs, filters, strides):
    """
    creates a residual block for resnet
    ref : https://miro.medium.com/max/792/1*7u6XWGYl7lLgc0EcKG1NMw.png
    :param inputs: input tensor
    :param filters: no of filters
    :param strides: strides
    :return: residual block constructor
    """
    shortcuts = inputs

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   strides=strides)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3,
                                   strides=strides)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcuts

    return inputs


def darknet53(inputs):
    """
    creates Darknet 53 model for feature extraction
    :param inputs:
    :return: C3, C4, C5 from stage 3, 4, 5 respectively as working similarly to feature extractor
    """
    inputs = conv2D_fiexed_padding(inputs,
                                   filters=32,
                                   kernel_size=3)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=64,
                                   kernel_size=3,
                                   strides=2)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 1st residual block
    inputs = darknet53_residualBlock(inputs,
                                     filters=32, strides=1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=128,
                                   kernel_size=3,
                                   strides=2)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 2nd residual block
    for _ in range(2):
        inputs = darknet53_residualBlock(inputs,
                                         filters=64,
                                         strides=1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=256,
                                   kernel_size=3,
                                   strides=2)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 3rd res block
    for _ in range(8):
        inputs = darknet53_residualBlock(inputs,
                                         filters=128,
                                         strides=1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=512,
                                   kernel_size=3,
                                   strides=2)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    C3 = inputs
    # 4th res block
    for _ in range(8):
        inputs = darknet53_residualBlock(inputs,
                                         filters=256,
                                         strides=1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=1024,
                                   kernel_size=3,
                                   strides=2)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    C4 = inputs
    # 5th res block
    for _ in range(4):
        inputs = darknet53_residualBlock(inputs,
                                         filters=512,
                                         strides=1)
    C5 = inputs

    return C3, C4, C5


def yoloConvBlock(inputs, filters):
    """
    Creates block for additional layer on the top of Yolo
    :param inputs: input image batch
    :param filters: no of filters
    :return: conv output
    """
    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    route = inputs

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3)
    inputs = batchNorm(inputs)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    return route, inputs
    pass


class darknet(tf.keras.Model):
    """
    Darkent model for feature extraction
    """

    def __init__(self, data_format):
        super().__init__()

        self.data_format = data_format

    def __call__(self):
        C3, C4, C5 = darknet53(self.inputs)

        # extract output from yolo layer (on top of the darknet)
        route, out = yoloConvBlock(C5,
                                   filters=512)

        out = conv2D_fiexed_padding(route,
                                    filters=256,
                                    kernel_size=1)
        out = batchNorm(out)
        out = tf.nn.leaky_relu(out, alpha=0.1)

        # concat C4 with out via upsampling
        upsample_size = C4.get_shape().as_list()
        out = upsample(out, out_shape=upsample_size, data_format=self.data_format)

        axis = 1 if self.data_format == 'channels_first' else 3
        out = tf.concat([out, C4], axis=axis)

        route, out = yoloConvBlock(out,
                                   filters=256)

        out = conv2D_fiexed_padding(route,
                                    filters=128,
                                    kernel_size=1)
        out = batchNorm(out)
        out = tf.nn.leaky_relu(out, alpha=0.1)

        # concat C4 with out via upsampling
        upsample_size = C3.get_shape().as_list()
        out = upsample(out, out_shape=upsample_size, data_format=self.data_format)

        axis = 1 if self.data_format == 'channels_first' else 3
        out = tf.concat([out, C3], axis=axis)

        route, out = yoloConvBlock(out,
                                   filters=128)

        return out

