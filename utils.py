import scipy.io as sio
import numpy as np
import os


class DataParser(object):

    def __init__(self, filepath='./data/vivo'):
        # raw_data: (counts=150, sensors=128, time_points=1524)
        # train_data: (130, 64, 1524)
        # valid_data: (10, 64, 1524)
        # test_data: (10, 64, 1524)
        self.path = filepath
        self.raw_data = []
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        files = os.listdir(filepath)
        for file in files:
            if file != '.DS_Store':
                self.load_data(os.path.join(filepath, file))
        self.clip_data()
        self.generate_data_set()

    def load_data(self, filename):
        mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        channels = mat['rf_channels']
        signals = [channels[:, :, i] for i in range(channels.shape[2])]
        self.raw_data.extend(signals)

    def clip_data(self):
        # clip data to shape (counts=150, sensors=128, time_points=1524)
        for idx, signal in enumerate(self.raw_data):
            self.raw_data[idx] = signal[:1524, :].transpose()

    def generate_data_set(self):
        for i in range(130):
            self.train_data.extend(self.raw_data[i])
        for i in range(130, 140):
            self.test_data.extend(self.raw_data[i])
        for i in range(140, 150):
            self.valid_data.extend(self.raw_data[i])

    def save(self):
        np.save(os.path.join(self.path, 'train.npy'), np.array(self.train_data))
        np.save(os.path.join(self.path, 'test.npy'), np.array(self.test_data))
        np.save(os.path.join(self.path, 'valid.npy'), np.array(self.valid_data))

    def merge(self, data_path_1, data_path_2):
        # data: (data_size * 64, 1524)
        # data file type: .npy
        data1 = np.load(data_path_1)
        data2 = np.load(data_path_2)
        data = np.concatenate([data1, data2], axis=0)
        return data


class DataGenerator(object):
    def __init__(self, data_type='all'):
        self.file_path = os.path.join('./data_norm', data_type)
        self.train = np.load(os.path.join(self.file_path, 'train.npy'))
        self.valid = np.load(os.path.join(self.file_path, 'valid.npy'))
        self.test = np.load(os.path.join(self.file_path, 'test.npy'))

    def generate_data(self, batch_size, num_step, data_set='train'):
        # train_data: (64 * ?, 1524)
        # valid_data: (64 * ?, 1524)
        # test_data: (64 * ?, 1524)
        # xs, ys: (?, batch_size, sensors, time_stamps)
        if data_set == 'test':
            data = self.test
        elif data_set == 'valid':
            data = self.valid
        else:
            data = self.train

        total_length = data.shape[0] // 2
        self.batches = total_length // (batch_size * num_step)

        assert (num_step <= 64 and 64 % num_step == 0), 'Invalid num_step: {}'.format(num_step)
        assert (batch_size * num_step <= total_length), 'Size out of range! Decrease the batch_size or num_step!'
        xx = data[::2, :]
        yy = data[1::2, :]
        xs = []
        ys = []
        for epoch in range(self.batches):
            x = []
            y = []
            for batch in range(batch_size):
                x.append(xx[batch * num_step: (batch + 1) * num_step, :])
                y.append(yy[batch * num_step: (batch + 1) * num_step, :])
            xs.append(x)
            ys.append(y)
        return xs, ys


if __name__ == '__main__':
    data = DataGenerator()
