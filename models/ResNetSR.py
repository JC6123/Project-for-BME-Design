from models.model import SR_Model
import tensorflow as tf
from Config import BaseConfig
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, \
    Activation
from tensorflow.python.keras import backend as K


class ResNetSRConfig(BaseConfig):
    def __init__(self):
        super(ResNetSRConfig, self).__init__()

        self.channels = 20

        self.n = 64
        self.nb_residual = 5

        self.mode = 'train'

        self.model_type = 'resnetsr'

    def __str__(self):
        return 'n-{}-channels-{}'.format(
            self.n, self.channels)


class ResNetSR_Model(SR_Model):
    def __init__(self, config):
        super(ResNetSR_Model, self).__init__(config)
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

        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(input_data)

        x1 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv2')(x0)
        x2 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x2, 1)
        for i in range(self.config.nb_residual):
            x = self._residual_block(x, i + 2)
        x = Add()([x, x2])

        x = self._upscale_block(x, 1)
        x = Add()([x, x1])

        x = self._upscale_block(x, 2)
        x = Add()([x, x0])

        output = Convolution2D(self.config.channels, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        prediction = tf.transpose(output, perm=[0, 3, 1, 2])
        prediction = tf.reshape(prediction, [batch_size, height, width])
        return prediction

    def _residual_block(self, ip, id):
        mode = True if self.config.mode == 'train' else False
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(self.config.n, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)

        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        # x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
        # x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = UpSampling2D()(init)
        x = Convolution2D(self.config.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        # x = Convolution2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
        #                            name='upsampling_deconv_%d' % id)(init)

        return x
