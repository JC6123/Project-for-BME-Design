from models.model import Model
from utils import DataGenerator
import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed, Dense, GRUCell, LSTMCell, SimpleRNNCell, StackedRNNCells, \
    Bidirectional, RNN
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import time
import numpy as np
import os


class Conv_BiRNN(Model):

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)
        self.loss_op = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss_op)
        self.summary_op = self.add_summary_op()

    def load_data(self):
        """
        Loads data from disk and stores it in memory.
        data[0], data[1]: (batch_size, num_steps, time_stamps)
        """
        data = DataGenerator(self.config.data_type)
        self.train_data = np.array(data.generate_data(self.config.batch_size, self.config.num_steps, data_set='train'))
        self.valid_data = np.array(data.generate_data(self.config.batch_size, self.config.num_steps, data_set='valid'))
        self.test_data = np.array(data.generate_data(self.config.batch_size, self.config.num_steps, data_set='test'))

    def add_placeholders(self):
        """
        Adds placeholder variables to tensorflow computational graph.

        """
        self.input_placeholder = tf.placeholder(
            shape=(self.config.batch_size, self.config.num_steps, self.config.time_stamps), dtype=tf.float64)

        self.output_placeholder = tf.placeholder(
            shape=(self.config.batch_size, self.config.num_steps, self.config.time_stamps), dtype=tf.float64)

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:

        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
              tensors created in add_placeholders.

        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.input_placeholder: input_batch}
        if label_batch is not None:
            feed_dict[self.output_placeholder] = label_batch
        return feed_dict

    def add_model(self, input_data):
        """Implements core of model that transforms input_data into predictions.

        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args:
          input_data: A tensor of shape (batch_size, num_steps, time_stamps).
        Returns:
          predict: A tensor of shape (batch_size, num_steps, time_stamps)
        """
        with tf.variable_scope('conv_layer'):
            # add dimension, input_data: (batch_size, num_steps, time_stamps, channels=1)
            input_data = tf.expand_dims(input_data, -1)

            max_filters = self.config.num_steps

            # conv1: (batch_size, W1, H1, F1=32)
            conv1 = Conv2D(filters=max_filters, kernel_size=(3, 3), padding='same', activation='relu')(input_data)
            # W2 = (W1 - F) / S + 1
            # H2 = (H1 - F) / S + 1
            # conv1: (batch_size, W2, H2, F1)
            conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv1 = tf.reshape(conv1, (self.config.batch_size, self.config.num_steps, -1))

        with tf.variable_scope('embedding_layer'):
            input_embeddings = TimeDistributed(
                Dense(self.config.rnn_size, dtype=tf.float64,
                      activation='tanh',
                      activity_regularizer=self.config.regularizer,
                      kernel_regularizer=self.config.regularizer,
                      bias_regularizer=self.config.regularizer),
                input_shape=(self.config.num_steps, self.config.time_stamps),
                dtype=tf.float64, activity_regularizer=self.config.regularizer)(
                conv1)
            # embeddings: (batch_size, num_steps, rnn_size)
            input_embeddings = tf.keras.layers.Dropout(self.config.dropout)(input_embeddings)

        with tf.variable_scope('rnn_layer'):
            if self.config.model_type == 'gru':
                cell_fun = GRUCell
            elif self.config.model_type == 'lstm':
                cell_fun = LSTMCell
            else:
                cell_fun = SimpleRNNCell

            cell = cell_fun(self.config.rnn_size, dtype=tf.float64, activity_regularizer=self.config.regularizer,
                            bias_regularizer=self.config.regularizer,
                            recurrent_regularizer=self.config.regularizer, kernel_regularizer=self.config.regularizer)
            cell = StackedRNNCells([cell] * self.config.num_layers)

            # outputs: (batch_size, num_steps * 2, rnn_size)
            outputs = Bidirectional(RNN(cell, return_sequences=True, dtype=tf.float64))(input_embeddings)

            # output: (batch_size * (num_steps * 2), rnn_size)
            output = tf.reshape(outputs, [-1, self.config.rnn_size])
            output = tf.keras.layers.Dropout(self.config.dropout)(output)

        with tf.variable_scope('output_layer'):
            # weights: (rnn_size, time_stamps)
            weights = tf.Variable(
                tf.truncated_normal([self.config.rnn_size, self.config.time_stamps], dtype=tf.float64, ),
                dtype=tf.float64)

            # bias: (time_stamps, )
            bias = tf.Variable(tf.zeros(shape=[self.config.time_stamps], dtype=tf.float64), dtype=tf.float64)

            # output: (batch_size * (num_steps * 2), time_stamps)
            predict_all = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
            predict_all = tf.nn.tanh(predict_all)
            predict_up = tf.slice(predict_all, [0, 0],
                                  [self.config.batch_size * self.config.num_steps, self.config.time_stamps])
            predict_down = tf.slice(predict_all, [self.config.batch_size * self.config.num_steps, 0],
                                    [self.config.batch_size * self.config.num_steps, self.config.time_stamps])

            predict = (predict_up + predict_down) / 2

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        return predict

    def add_loss_op(self, pred):
        """Adds ops for loss to the computational graph.

        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar) output
        """
        # labels: (batch_size * (num_steps * 2), time_stamps)
        labels = tf.reshape(self.output_placeholder, [-1, self.config.time_stamps])
        loss = tf.reduce_mean(tf.squared_difference(pred, labels))
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss_op,
                                                                              global_step=self.global_step)
        return train_op

    def add_summary_op(self):
        tf.summary.scalar('loss', self.loss_op)
        summary_op = tf.summary.merge_all()
        return summary_op

    def run_epoch(self, sess, input_data, input_labels):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
          sess: tf.Session() object
          input_data: (batch_count, batch_size, num_step, time_stamps)
          input_labels: (batch_count, batch_size, num_step, time_stamps)
        Returns:
          average_loss: scalar. Average minibatch loss of model on epoch.
        """
        train_average_loss = 0
        batch_count = input_data.shape[0]

        for batch in range(batch_count):
            train_feed_dict = self.create_feed_dict(input_data[batch], input_labels[batch])
            train_loss, _, summary = sess.run([self.loss_op, self.train_op, self.summary_op], feed_dict=train_feed_dict)
            self.train_writer.add_summary(summary, global_step=self.global_step.eval())
            train_average_loss += train_loss
        train_average_loss /= batch_count
        return train_average_loss

    def fit(self, sess, input_data, input_labels):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: (batch_count, batch_size, num_step, time_stamps)
          input_labels: (batch_count, batch_size, num_step, time_stamps)
        Returns:
          losses: list of loss per epoch
        """

        self.train_writer = tf.summary.FileWriter(self.config.summary_dir + '-train', sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.config.summary_dir + '-valid', sess.graph)
        self.saver = tf.train.Saver(tf.global_variables())

        start_epoch = 0

        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        if checkpoint:
            self.saver.restore(sess, checkpoint)
            print("==================== Restore from the checkpoint {0} ====================".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])

        train_losses = []
        valid_losses = []
        with sess:
            try:
                for epoch in range(start_epoch, self.config.epochs):
                    start_time = time.time()

                    train_average_loss = self.run_epoch(sess, input_data, input_labels)

                    valid_average_loss = self.valid(sess, self.test_data[0], self.test_data[1])

                    duration = time.time() - start_time
                    print('Epoch %d: train_loss = %f,  valid_loss = %f (%.3f sec)'
                          % (epoch, train_average_loss, valid_average_loss, duration))

                    self.saver.save(sess, os.path.join(self.config.checkpoints_dir, self.config.model_prefix),
                                    global_step=epoch)

                    train_losses.append(train_average_loss)
                    valid_losses.append(valid_average_loss)
            except KeyboardInterrupt:
                print('Interrupt manually, try saving checkpoint for now...')
                self.saver.save(sess, os.path.join(self.config.checkpoints_dir, self.config.model_prefix),
                                global_step=epoch)
                print(
                    '====================Last epoch were saved, next time will start from epoch {}.===================='.format(
                        epoch))
        return train_losses, valid_losses

    def valid(self, sess, input_data, input_labels=None):
        """Make valid from the provided model.
        Args:
          sess: tf.Session()
          input_data: (1, max_batch_size, num_step, time_stamps)
          input_labels: (1, max_batch_size, num_step, time_stamps)
        Returns:
          average_loss: Average loss of model.
        """
        valid_average_loss = 0
        batch_count = input_data.shape[0]

        for batch in range(batch_count):
            valid_feed_dict = self.create_feed_dict(input_data[batch], input_labels[batch])
            valid_loss, _, summary = sess.run([self.loss_op, self.train_op, self.summary_op], feed_dict=valid_feed_dict)
            self.valid_writer.add_summary(summary, global_step=self.global_step.eval())
            valid_average_loss += valid_loss
        valid_average_loss /= batch_count
        return valid_average_loss
