import tensorflow as tf
from models.model import SR_Model

from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras import Model
from Config import BaseConfig

from random import random


class Seq2SeqConfig(BaseConfig):
    def __init__(self):
        super(Seq2SeqConfig, self).__init__()
        self.enc_units = 1024
        self.dec_units = 1024

        self.model_type = 'seq2seq_rdm_teaching'
        self.teaching = False

    def __str__(self):
        return 'enc_units-{}-dec_units-{}'.format(
            self.enc_units,
            self.dec_units)


class Encoder(Model):
    """
    Encoder output shape: (batch_size, sequence_length, units)
    Encoder hidden state: (batch_size, units)
    """

    def __init__(self, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initial_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_t, hiddens):
        # hiddens: (batch_size, sequence_length, hidden_size)
        # hidden_t: (batch_size, hidden_size)
        # hidden_with_time_axis: (batch_size, 1, hidden_size)
        hidden_t_with_time_axis = tf.expand_dims(hidden_t, axis=1)

        # score: (batch_size, sequence_length, 1)
        score = self.V(tf.nn.tanh(self.W1(hiddens) + self.W2(hidden_t_with_time_axis)))

        # attention_weights: (batch_size, sequence_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector: (batch_size, hidden_size)
        context_vector = attention_weights * hiddens
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(Model):
    def __init__(self, time_stamps, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc = Dense(time_stamps)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x: (batch_size, 1, time_stamps)
        # output: (batch_size, 1, time_stamps)
        # enc_output: (batch_size, sequence_length, hidden_size)
        # hidden: (batch_size, hidden_size)

        # context_vector: (batch_size, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x: (batch_size, 1, time_stamps + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=2)

        output, state = self.gru(x)

        # output: (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output: (batch_size, time_stamps)
        output = self.fc(output)

        # output: (batch_size, 1, time_stamps)
        output = tf.expand_dims(output, axis=1)
        return output, state, attention_weights


class Seq2Seq_Model(SR_Model):
    def __init__(self, config):
        super(Seq2Seq_Model, self).__init__(config)

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
        encoder = Encoder(self.config.enc_units, self.config.batch_size)
        decoder = Decoder(self.config.time_stamps, self.config.dec_units, self.config.batch_size)

        enc_hidden = encoder.initial_hidden_state()
        enc_output, enc_hidden = encoder(input_data, enc_hidden)

        dec_hidden = enc_hidden
        predictions = []

        # Teacher forcing
        dec_input = tf.expand_dims(input_data[:, 0, :], axis=1)
        for t in range(self.config.num_steps):
            # pred: (batch_size, 1, time_stamps)
            pred, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            predictions.append(pred)
            if t == self.config.num_steps - 1:
                break
            if random() > 0.5:
                teaching = True
            else:
                teaching = False
            if self.config.teaching == 0:
                dec_input = tf.expand_dims(target_data[:, t, :], axis=1)
            elif self.config.teaching == 1:
                dec_input = pred
            else:
                if teaching:
                    dec_input = tf.expand_dims(target_data[:, t, :], axis=1)
                else:
                    dec_input = pred
        # predictions: (batch_size, num_steps, time_stamps)
        predictions = tf.concat(predictions, axis=1)

        return predictions
