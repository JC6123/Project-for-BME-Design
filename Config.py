import tensorflow as tf
from tensorflow.python.keras.regularizers import l1, l2, l1_l2


class BaseConfig(object):
    def __init__(self):
        self.batch_size = 15
        self.learning_rate = 0.0001
        self.epochs = 6000

        self.time_stamps = 1524
        self.num_steps = 64
        self.dropout = 0.5

        self.regularizer_type = 'l2'
        if self.regularizer_type == 'l2':
            self.regularizer = l2(1)
        elif self.regularizer_type == 'l1':
            self.regularizer = l1(1)
        elif self.regularizer == 'l1_l2':
            self.regularizer = l1_l2(1)

        self.model_type = ''
        self.data_type = 'all'


    @property
    def checkpoints_dir(self):
        return 'checkpoints/SR-{}-{}-{}'.format(self.data_type, self.model_type, self.__str__())

    @property
    def summary_dir(self):
        return 'graphs/SR-{}-{}-{}'.format(self.data_type, self.model_type, self.__str__())

    @property
    def model_prefix(self):
        return 'SR-{}-{}-{}'.format(self.data_type, self.model_type, self.__str__())

    def __str__(self):
        return 'batch_size-{}-num_steps-{}-regulator-{}-dropout-{}'.format(
            self.batch_size,
            self.num_steps,
            self.regularizer_type,
            self.dropout, )
