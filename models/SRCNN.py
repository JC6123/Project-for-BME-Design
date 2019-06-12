from models.model import SR_Model
import tensorflow as tf
from Config import BaseConfig
from tensorflow.python.keras.layers import Convolution2D


class SRCNNConfig(BaseConfig):
    def __init__(self):
        super(SRCNNConfig, self).__init__()

        self.channels = 10

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32
        self.model_type = 'srcnn'

    def __str__(self):
        return 'f3-{}-{}-{}-n-{}-{}-channels-{}'.format(
            self.f1, self.f2, self.f3, self.n1, self.n2, self.channels)


class SRCNN_Model(SR_Model):
    def __init__(self, config):
        super(SRCNN_Model, self).__init__(config)
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

        # conv-1    x: (-1, height, width, n1)
        x = Convolution2D(self.config.n1, (self.config.f1, self.config.f1),
                          activation='relu',
                          padding='same',
                          name='conv-1',
                          kernel_regularizer=self.config.regularizer,
                          bias_regularizer=self.config.regularizer,
                          )(input_data)

        # conv-2    x: (-1, height, width, n2)
        x = Convolution2D(self.config.n2, (self.config.f2, self.config.f2),
                          activation='relu',
                          padding='same',
                          name='conv-2',
                          kernel_regularizer=self.config.regularizer,
                          bias_regularizer=self.config.regularizer,
                          )(x)

        # conv-3    x: (-1, height, width, n2)
        x = Convolution2D(self.config.n2, (self.config.f2, self.config.f2),
                          activation='relu',
                          padding='same',
                          name='conv-3',
                          kernel_regularizer=self.config.regularizer,
                          bias_regularizer=self.config.regularizer,
                          )(x)

        # conv-4    x: (-1, height, width, n2)
        x = Convolution2D(self.config.n2, (self.config.f2, self.config.f2),
                          activation='relu',
                          padding='same',
                          name='conv-4',
                          kernel_regularizer=self.config.regularizer,
                          bias_regularizer=self.config.regularizer,
                          )(x)

        # output    x: (-1, height, width, channels)
        output = Convolution2D(channels, (self.config.f3, self.config.f3),
                               activation='linear',
                               padding='same',
                               name='output',
                               kernel_regularizer=self.config.regularizer,
                               bias_regularizer=self.config.regularizer,
                               )(x)

        prediction = tf.transpose(output, perm=[0, 3, 1, 2])
        prediction = tf.reshape(prediction, [batch_size, height, width])
        return prediction
