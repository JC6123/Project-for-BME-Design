from models.BiRNN import BiRNN_Model
from models.Seq2Seq import Seq2Seq_Model
import tensorflow as tf
from Config import BiRNNConfig
from Config import Seq2SeqConfig
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def train(config):
    model = BiRNN_Model(config)
    init = tf.global_variables_initializer()
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess = tf.Session()
    sess.run(init)
    losses = model.fit(sess, model.train_data[0], model.train_data[1])


def predict(mode='test'):
    config = BiRNNConfig()
    model = BiRNN_Model(config)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if mode == 'train':
        low_resolution = model.train_data[0]
        ground_truths = model.train_data[1]
    elif mode == 'test':
        low_resolution = model.test_data[0]
        ground_truths = model.test_data[1]
    else:
        low_resolution = model.valid_data[0]
        ground_truths = model.valid_data[1]

    prediction = model.predict(sess, low_resolution, ground_truths)

    return low_resolution, prediction, ground_truths


def test():
    l, p, t = predict('test')

    prediction = p.reshape((-1, 1527))
    truth = t.reshape((-1, 1527))
    low_resolution = l.reshape((-1, 1527))

    MSE = np.mean((prediction - truth) ** 2)
    ER = np.mean(np.abs(prediction - truth)) / np.mean(np.abs(truth)) * 100

    indices = range(low_resolution.shape[0])
    high_resolution_prediction = np.insert(prediction, indices, low_resolution, axis=0).reshape((-1, 128, 1527))
    high_resolution_ground_truths = np.insert(truth, indices, low_resolution, axis=0).reshape((-1, 128, 1527))

    np.save('tests/HR_prediction-gru-1024.npy', high_resolution_prediction)
    sio.savemat('tests/HR_prediction-gru-1024.mat', {'data': high_resolution_prediction})
    np.save('tests/HR_truth.npy', high_resolution_ground_truths)
    sio.savemat('tests/truths.mat', {'data': high_resolution_ground_truths})

    print('MSE: {}, ER: {} %'.format(MSE, ER))


def draw(file):
    model_name = file.split('/')[1].split('.')[0]
    truth = np.load('tests/HR_truth.npy')
    pred = np.load(file)

    plt.plot(truth[0][125][:500], color='red', linewidth=0.5, linestyle='-', label='truth')
    plt.plot(pred[0][125][:500], color='blue', linewidth=0.5, linestyle='-', label='prediction')

    plt.legend(loc='upper left')
    plt.title(model_name, fontsize='large', fontweight='bold')

    plt.savefig('tests/{}.png'.format(model_name), dpi=300)


if __name__ == '__main__':
    test()
    draw('tests/HR_prediction-gru-1024.npy')
    # train(BiRNNConfig())
