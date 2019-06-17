import math
import numpy as np
import tensorflow as tf
from enum import Enum, unique


@unique
class InputType(Enum):
    TENSOR = 1
    BASE64_JPEG = 2


class OpenNsfwModel:
    """Tensorflow implementation of Yahoo's Open NSFW Model

    Original implementation:
    https://github.com/yahoo/open_nsfw

    Weights have been converted using caffe-tensorflow:
    https://github.com/ethereon/caffe-tensorflow
    """

    def __init__(self):
        self.weights = {}
        self.bn_epsilon = 1e-5  # Default used by Caffe

    def build(self, weights_path="open_nsfw-weights.npy",
              input_type=InputType.TENSOR):

        self.weights = np.load(weights_path, encoding="latin1").item()
        self.input_tensor = None

        if input_type == InputType.TENSOR:
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, 224, 224, 3],
                                        name="input")
            self.input_tensor = self.input
        elif input_type == InputType.BASE64_JPEG:
            from image_utils import load_base64_tensor

            self.input = tf.placeholder(tf.string, shape=(None,), name="input")
            self.input_tensor = load_base64_tensor(self.input)
        else:
            raise ValueError("invalid input type '{}'".format(input_type))

        x = self.input_tensor

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = self.__conv2d("conv_1", x, filter_depth=64,
                          kernel_size=7, stride=2, padding='valid')

        x = self.__batch_norm("bn_1", x)
        x = tf.nn.relu(x)

        x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

        x = self.__conv_block(stage=0, block=0, inputs=x,
                              filter_depths=[32, 32, 128],
                              kernel_size=3, stride=1)

        x = self.__identity_block(stage=0, block=1, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)
        x = self.__identity_block(stage=0, block=2, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)

        x = self.__conv_block(stage=1, block=0, inputs=x,
                              filter_depths=[64, 64, 256],
                              kernel_size=3, stride=2)
        x = self.__identity_block(stage=1, block=1, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=2, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=3, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)

        x = self.__conv_block(stage=2, block=0, inputs=x,
                              filter_depths=[128, 128, 512],
                              kernel_size=3, stride=2)
        x = self.__identity_block(stage=2, block=1, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=2, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=3, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=4, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=5, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)

        x = self.__conv_block(stage=3, block=0, inputs=x,
                              filter_depths=[256, 256, 1024], kernel_size=3,
                              stride=2)
        x = self.__identity_block(stage=3, block=1, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)
        x = self.__identity_block(stage=3, block=2, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)

        x = tf.layers.average_pooling2d(x, pool_size=7, strides=1,
                                        padding="valid", name="pool")

        x = tf.reshape(x, shape=(-1, 1024))

        self.logits = self.__fully_connected(name="fc_nsfw",
                                             inputs=x, num_outputs=2)
        self.predictions = tf.nn.softmax(self.logits, name="predictions")

    """Get weights for layer with given name
    """
    def __get_weights(self, layer_name, field_name):
        if not layer_name in self.weights:
            raise ValueError("No weights for layer named '{}' found"
                             .format(layer_name))

        w = self.weights[layer_name]
        if not field_name in w:
            raise (ValueError("No entry for field '{}' in layer named '{}'"
                              .format(field_name, layer_name)))

        return w[field_name]

    """Layer creation and weight initialization
    """
    def __fully_connected(self, name, inputs, num_outputs):
        return tf.layers.dense(
            inputs=inputs, units=num_outputs, name=name,
            kernel_initializer=tf.constant_initializer(
                self.__get_weights(name, "weights"), dtype=tf.float32),
            bias_initializer=tf.constant_initializer(
                self.__get_weights(name, "biases"), dtype=tf.float32))

    def __conv2d(self, name, inputs, filter_depth, kernel_size, stride=1,
                 padding="same", trainable=False):

        if padding.lower() == 'same' and kernel_size > 1:
            if kernel_size > 1:
                oh = inputs.get_shape().as_list()[1]
                h = inputs.get_shape().as_list()[1]

                p = int(math.floor(((oh - 1) * stride + kernel_size - h)//2))

                inputs = tf.pad(inputs,
                                [[0, 0], [p, p], [p, p], [0, 0]],
                                'CONSTANT')
            else:
                raise Exception('unsupported kernel size for padding: "{}"'
                                .format(kernel_size))

        return tf.layers.conv2d(
            inputs, filter_depth,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), padding='valid',
            activation=None, trainable=trainable, name=name,
            kernel_initializer=tf.constant_initializer(
                self.__get_weights(name, "weights"), dtype=tf.float32),
            bias_initializer=tf.constant_initializer(
                self.__get_weights(name, "biases"), dtype=tf.float32))

    def __batch_norm(self, name, inputs, training=False):
        return tf.layers.batch_normalization(
            inputs, training=training, epsilon=self.bn_epsilon,
            gamma_initializer=tf.constant_initializer(
                self.__get_weights(name, "scale"), dtype=tf.float32),
            beta_initializer=tf.constant_initializer(
                self.__get_weights(name, "offset"), dtype=tf.float32),
            moving_mean_initializer=tf.constant_initializer(
                self.__get_weights(name, "mean"), dtype=tf.float32),
            moving_variance_initializer=tf.constant_initializer(
                self.__get_weights(name, "variance"), dtype=tf.float32),
            name=name)

    """ResNet blocks
    """
    def __conv_block(self, stage, block, inputs, filter_depths,
                     kernel_size=3, stride=2):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths

        conv_name_base = "conv_stage{}_block{}_branch".format(stage, block)
        bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)
        shortcut_name_post = "_stage{}_block{}_proj_shortcut" \
                             .format(stage, block)

        shortcut = self.__conv2d(
            name="conv{}".format(shortcut_name_post), stride=stride,
            inputs=inputs, filter_depth=filter_depth3, kernel_size=1,
            padding="same"
        )

        shortcut = self.__batch_norm("bn{}".format(shortcut_name_post),
                                     shortcut)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=stride, padding="same",
        )
        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, shortcut)

        return tf.nn.relu(x)

    def __identity_block(self, stage, block, inputs,
                         filter_depths, kernel_size):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths
        conv_name_base = "conv_stage{}_block{}_branch".format(stage, block)
        bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=1, padding="same",
        )

        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, inputs)

        return tf.nn.relu(x)
