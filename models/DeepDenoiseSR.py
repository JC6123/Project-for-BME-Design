from models.model import SR_Model
import tensorflow as tf
from Config import BaseConfig
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Add


class DeepDenoiseSRConfig(BaseConfig):
    def __init__(self):
        super(DeepDenoiseSRConfig, self).__init__()

        self.channels = 20

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.model_type = 'ddsrcnn'

    def __str__(self):
        return 'n-{}-{}-{}-channels-{}'.format(
            self.n1, self.n2, self.n3, self.channels)


class DeepDenoiseSR_Model(SR_Model):
    def __init__(self, config):
        super(DeepDenoiseSR_Model, self).__init__(config)
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

        c1 = Convolution2D(self.config.n1, (3, 3), activation='relu', padding='same')(input_data)
        c1 = Convolution2D(self.config.n1, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.config.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Convolution2D(self.config.n2, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.config.n3, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.config.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Convolution2D(self.config.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.config.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Convolution2D(self.config.n1, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        output = Convolution2D(channels, (5, 5), activation='linear', padding='same')(m2)

        prediction = tf.transpose(output, perm=[0, 3, 1, 2])
        prediction = tf.reshape(prediction, [batch_size, height, width])
        return prediction
