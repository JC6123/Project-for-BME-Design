from utils import DataGenerator
import tensorflow as tf
import time
import numpy as np
import os
import logging


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def load_data(self):
        """Loads data from disk and stores it in memory.

        Feel free to add instance variables to Model object that store loaded data.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building code and will be fed data during
        training.

        See for more information:

        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

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
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_model(self, input_data, target_data=None):
        """Implements core of model that transforms input_data into predictions.

        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args:
          input_data: A tensor of shape (batch_size, n_features).
          target_data: A tensor of shape (batch_size, n_features).
        Returns:
          out: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds ops for loss to the computational graph.

        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess, input_data, input_labels):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
          sess: tf.Session() object
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: scalar. Average minibatch loss of model on epoch.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def fit(self, sess, input_data, input_labels):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def predict(self, sess, input_data):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
        Returns:
          predictions: Predictions of model on input_data
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
              Calling optimizer.minimize() will return a train_op object.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_summary_op(self):
        """
        Sets up summary Ops.
        :return: summary_op
        """


class SR_Model(Model):
    def __init__(self, config):
        self.config = config
        logging.basicConfig(filename=os.path.join('logs', self.config.model_prefix + '.log'),
                            level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filemode='w'
                            )
        self.load_data()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder, self.output_placeholder)
        self.loss_op = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss_op)
        self.summary_op = self.add_summary_op()

    def load_data(self):
        """
        Loads data from disk and stores it in memory.
        data[0], data[1]: (?, batch_size, num_steps, time_stamps)
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
            shape=(self.config.batch_size, self.config.num_steps, self.config.time_stamps), dtype=tf.float32)

        self.output_placeholder = tf.placeholder(
            shape=(self.config.batch_size, self.config.num_steps, self.config.time_stamps), dtype=tf.float32)

    def create_feed_dict(self, input_batch, label_batch=None):
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

    def add_loss_op(self, pred):
        """Adds ops for loss to the computational graph.

        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar) output
        """
        # labels: (batch_size * (num_steps * 2), time_stamps)
        labels = tf.reshape(self.output_placeholder, [-1, self.config.time_stamps])
        pred = tf.reshape(pred, [-1, self.config.time_stamps])
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
        """
        logging.info("Training start: {}".format(self.config.model_prefix))

        self.train_writer = tf.summary.FileWriter(self.config.summary_dir + '-train', sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.config.summary_dir + '-valid', sess.graph)
        self.saver = tf.train.Saver(tf.global_variables())

        start_epoch = 0

        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        if checkpoint:
            self.saver.restore(sess, checkpoint)
            logging.info("==================== Restore from the checkpoint {0} ====================".format(checkpoint))
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
                    logging.info('Epoch %d: train_loss = %f,  valid_loss = %f (%.3f sec)'
                                 % (epoch, train_average_loss, valid_average_loss, duration))

                    self.saver.save(sess, os.path.join(self.config.checkpoints_dir, self.config.model_prefix),
                                    global_step=epoch)

                    train_losses.append(train_average_loss)
                    valid_losses.append(valid_average_loss)
            except KeyboardInterrupt:
                logging.info('Interrupt manually, try saving checkpoint for now...')
                self.saver.save(sess, os.path.join(self.config.checkpoints_dir, self.config.model_prefix),
                                global_step=epoch)
                logging.info(
                    '====================Last epoch were saved, next time will start from epoch {}.===================='.format(
                        epoch))

    def valid(self, sess, input_data, input_labels=None):
        """Make valid from the provided model.
        Args:
          sess: tf.Session()
          input_data: (n, batch_size, num_step, time_stamps)
          input_labels: (n, batch_size, num_step, time_stamps)
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

    def predict(self, sess, input_data, input_labels=None):
        """
        Make prediction from the provided model.
        :param sess: tf.Session()
        :param input_data: (n, batch_size, num_step, time_stamps)
        :param input_labels: (n, batch_size, num_step, time_stamps)
        :return: prediction: (n, batch_size, num_step, time_stamps)
                 ground_truths: (n, batch_size, num_step, time_stamps)
        """
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        if checkpoint:
            self.saver.restore(sess, checkpoint)
            print("==================== Restore from the checkpoint {0} ====================".format(checkpoint))
        else:
            print("No existed model!")
            return None, None

        batch_count = input_data.shape[0]
        prediction = []

        for batch in range(batch_count):
            feed_dict = self.create_feed_dict(input_data[batch], input_labels[batch])
            pred = sess.run([self.pred], feed_dict=feed_dict)
            prediction.append(pred)
        return np.array(prediction)
