from models.model import SR_Model
import tensorflow as tf
from Config import BaseConfig
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import concatenate


def layer_conv_relu(layer_input, filters, nb_row, nb_col, nb_regularizer):
    layer_process = Convolution2D(filters, (nb_row, nb_col), activation='relu', padding='same',
                                  kernel_regularizer=nb_regularizer,
                                  bias_regularizer=nb_regularizer,
                                  data_format='channels_last',
                                  )(layer_input)
    layer_output = Activation('relu')(layer_process)
    return layer_output


class UISRCNN_kso_Config(BaseConfig):
    def __init__(self):
        super(UISRCNN_kso_Config, self).__init__()

        self.channels = 1

        self.f1 = 3
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 128
        self.model_type = 'uisrcnn_kso'

    def __str__(self):
        return '27layers-{}-{}-n-{}-{}-channels-{}'.format(
            self.f1, self.f2, self.n1, self.n2, self.channels)


class UISRCNN_kso_Model(SR_Model):
    def __init__(self, config):
        super(UISRCNN_kso_Model, self).__init__(config)
        assert self.config.batch_size % self.config.channels == 0, 'Invalid pair of batch size({}) and channels({})'.format(
            self.config.batch_size, self.config.channels)

    def add_model(self, input_data, target_data=None):
        """Implements core of model that transforms input_data into predictions.

        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args:
          input_data: A tensor of shape (batch_size, num_steps, time_stamps).
          target_data: A tensor of shape (batch_size, num_steps, time_stamps).
        Returns:
          predict: A tensor of shape (batch_size, num_steps, time_stamps)
        """
        # Consider signal matrix as an image with channels.
        height = self.config.num_steps
        width = self.config.time_stamps
        channels = self.config.channels
        batch_size = self.config.batch_size

        # input_data: (-1, height, width, channels)
        input_data = tf.reshape(input_data, [-1, channels, height, width])
        input_data = tf.transpose(input_data, perm=[0, 2, 3, 1])

        # module-1
        # conv/ReLU-1
        x = layer_conv_relu(input_data, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x_1 = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-2
        # conv/ReLU-1
        x = layer_conv_relu(x_1, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x_2 = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-3
        # conv/ReLU-1
        x = layer_conv_relu(x_2, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x_3 = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-4
        # conv/ReLU-1
        x = layer_conv_relu(x_3, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x_4 = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-5
        # conv/ReLU-1
        x = layer_conv_relu(x_4, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-6
        # conv/ReLU-1
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # concatenate
        x = concatenate([x, x_4], axis=-1)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-7
        # conv/ReLU-1
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # concatenate
        x = concatenate([x, x_3], axis=-1)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-8
        # conv/ReLU-1
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # concatenate
        x = concatenate([x, x_2], axis=-1)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/ReLU-3
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)

        # module-9
        # conv/ReLU-1
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # concatenate
        x = concatenate([x, x_1], axis=-1)
        # conv/ReLU-2
        x = layer_conv_relu(x, self.config.n1, self.config.f1, self.config.f3, self.config.regularizer)
        # conv/tanh-3
        x = Convolution2D(self.config.n1, (self.config.f1, self.config.f3),
                          activation='relu', padding='same',
                          kernel_regularizer=self.config.regularizer,
                          bias_regularizer=self.config.regularizer,
                          data_format='channels_last',
                          )(x)
        x = Activation('tanh')(x)

        # output    x: (-1, height, width, channels)
        output = Convolution2D(channels, (self.config.f2, self.config.f2),
                               activation='linear',
                               padding='same',
                               name='output',
                               kernel_regularizer=self.config.regularizer,
                               bias_regularizer=self.config.regularizer,
                               data_format='channels_last'
                               )(x)

        prediction = tf.transpose(output, perm=[0, 3, 1, 2])
        prediction = tf.reshape(prediction, [batch_size, height, width])
        return prediction
