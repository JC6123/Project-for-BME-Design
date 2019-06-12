from models.model import SR_Model
import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed, Dense, GRUCell, LSTMCell, SimpleRNNCell, StackedRNNCells, \
    Bidirectional, RNN
from Config import BaseConfig


class BiRNNConfig(BaseConfig):
    def __init__(self):
        super(BiRNNConfig, self).__init__()
        self.rnn_size = 1024
        self.num_layers = 2

        self.model_type = 'gru'

    def __str__(self):
        return 'rnn_size-{}-num_layers-{}'.format(
            self.rnn_size,
            self.num_layers)


class BiRNN_Model(SR_Model):

    def __init__(self, config):
        super(BiRNN_Model, self).__init__(config)

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
        with tf.variable_scope('embedding_layer'):
            input_embeddings = TimeDistributed(
                Dense(self.config.rnn_size, dtype=tf.float32, activity_regularizer=self.config.regularizer,
                      kernel_regularizer=self.config.regularizer,
                      bias_regularizer=self.config.regularizer),
                input_shape=(self.config.num_steps, self.config.time_stamps),
                dtype=tf.float32, activity_regularizer=self.config.regularizer)(
                self.input_placeholder)
            input_embeddings = tf.keras.layers.Dropout(self.config.dropout)(input_embeddings)

        with tf.variable_scope('rnn_layer'):
            if self.config.model_type == 'gru':
                cell_fun = GRUCell
            elif self.config.model_type == 'lstm':
                cell_fun = LSTMCell
            else:
                cell_fun = SimpleRNNCell

            cell = cell_fun(self.config.rnn_size, dtype=tf.float32, activity_regularizer=self.config.regularizer,
                            bias_regularizer=self.config.regularizer,
                            recurrent_regularizer=self.config.regularizer, kernel_regularizer=self.config.regularizer)
            cell = StackedRNNCells([cell] * self.config.num_layers)

            # outputs: (batch_size, num_steps * 2, rnn_size)
            outputs = Bidirectional(RNN(cell, return_sequences=True, dtype=tf.float32))(input_embeddings)

            # output: (batch_size * (num_steps * 2), rnn_size)
            output = tf.reshape(outputs, [-1, self.config.rnn_size])
            output = tf.keras.layers.Dropout(self.config.dropout)(output)

        with tf.variable_scope('output_layer'):
            # weights: (rnn_size, time_stamps)
            weights = tf.Variable(
                tf.truncated_normal([self.config.rnn_size, self.config.time_stamps], dtype=tf.float32, ),
                dtype=tf.float32)

            # bias: (time_stamps, )
            bias = tf.Variable(tf.zeros(shape=[self.config.time_stamps], dtype=tf.float32), dtype=tf.float32)

            # output: (batch_size * (num_steps * 2), time_stamps)
            predict_all = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
            predict_up = tf.slice(predict_all, [0, 0],
                                  [self.config.batch_size * self.config.num_steps, self.config.time_stamps])
            predict_down = tf.slice(predict_all, [self.config.batch_size * self.config.num_steps, 0],
                                    [self.config.batch_size * self.config.num_steps, self.config.time_stamps])

            predict = (predict_up + predict_down) / 2

        return predict
