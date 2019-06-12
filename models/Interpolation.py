from utils import DataGenerator
from scipy.interpolate import interp2d
import numpy as np


def mse(prediction, truth):
    return np.mean((prediction - truth) ** 2)


def interpolation(kind='cubic'):
    data = DataGenerator('all').valid
    x_length = data.shape[1]
    y_length = int(data.shape[0] / 2)
    test_data = data[::2, :]
    ground_truth = data[1::2, :]
    f = interp2d(np.arange(x_length), np.arange(y_length), test_data, kind=kind)

    prediction = f(np.arange(x_length), np.arange(1, y_length + 1))
    zeros = np.zeros(shape=test_data.shape)
    # print('zero baseline = {}'.format(mse(zeros, ground_truth)))
    print(kind + '-interpolation MSE loss = {}'.format(mse(prediction, ground_truth)))


if __name__ == '__main__':
    interpolation_kinds = ['linear', 'cubic', 'quintic']
    for kind in interpolation_kinds:
        interpolation(kind)
