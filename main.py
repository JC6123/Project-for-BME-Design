import argparse
from models.BiRNN import BiRNN_Model, BiRNNConfig
from models.Seq2Seq import Seq2Seq_Model, Seq2SeqConfig
from models.SRCNN import SRCNN_Model, SRCNNConfig
from models.DeepDenoiseSR import DeepDenoiseSR_Model, DeepDenoiseSRConfig
from models.ResNetSR import ResNetSR_Model, ResNetSRConfig
from models.Ultrasound_Image_SRCNN import UISRCNN_Model, UISRCNNConfig
from models.UISRCNN_kernel_size_optimized import UISRCNN_kso_Config, UISRCNN_kso_Model
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import scipy.io as sio
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser("Train SR model.")

parser.add_argument('--model', default='uisrcnn_kso', type=str, dest='model_type',
                    help='Model type: "birnn", "seq2seq", "srcnn", "ddsrcnn", "resnetsr", '
                         '"uisrcnn", "uisrcnn_kso"'
                    )
parser.add_argument('--mode', default='train', type=str, dest='mode', help='Mode: train or test.')

parser.add_argument('--rnn_cell', default='gru', type=str, dest='rnn_cell',
                    help='Rnn cell type: "rnn", "lstm" or "gru".')
parser.add_argument('--rnn_layers', default=2, type=int, dest='rnn_layers', help='RNN layers for BiRNN model.')
parser.add_argument('--hidden_size', default=64, type=int, dest='hidden_size', help='Hidden layer size.')

parser.add_argument('--teach', type=str, dest='teach',
                    help='Force teaching argument for seq2seq model: "yes", "no", or "rdm".')

parser.add_argument('--channels', type=int, default=20, dest='channels', help='Channels number for image model.')
parser.add_argument('--filters', type=int, default=64, dest='conv_filters', help='Filter number of convolution layer.')
parser.add_argument('--block1', type=int, default=5, dest='block1', help='Block1 number of deep convolution network.')

args = parser.parse_args()


def create_model():
    if args.model_type not in ('birnn', 'seq2seq', 'srcnn', 'ddsrcnn', 'resnetsr', 'uisrcnn', 'uisrcnn_kso'):
        raise ValueError('Model type must be either "birnn", "seq2seq", "srcnn", "ddsrcnn", "resnetsr", '
                         '"uisrcnn", "uisrcnn_kso".'
                         )

    if args.mode not in ('train', 'test'):
        raise ValueError('Mode must be either "train" or "test".')

    if args.model_type == 'birnn':
        config = BiRNNConfig()
        config.rnn_size = args.hidden_size
        config.model_type = args.rnn_cell
        config.num_layers = args.rnn_layers
        model = BiRNN_Model(config)
    elif args.model_type == 'seq2seq':
        config = Seq2SeqConfig()
        config.enc_units = args.hidden_size
        config.dec_units = args.hidden_size
        if args.teach == 'yes':
            config.model_type = 'seq2seq_teaching'
            config.teaching = 1
        elif args.teach == 'no':
            config.model_type = 'seq2seq_no_teaching'
            config.teaching = 0
        elif args.teach == 'rdm':
            config.model_type = 'seq2seq_rdm_teaching'
            config.teaching = -1

        model = Seq2Seq_Model(config)
    elif args.model_type == 'srcnn':
        config = SRCNNConfig()
        config.channels = args.channels
        model = SRCNN_Model(config)
    elif args.model_type == 'ddsrcnn':
        config = DeepDenoiseSRConfig()
        config.channels = args.channels
        model = DeepDenoiseSR_Model(config)
    elif args.model_type == 'resnetsr':
        config = ResNetSRConfig()
        config.channels = args.channels
        config.n = args.conv_filters
        config.nb_residual = args.block1
        model = ResNetSR_Model(config)
    elif args.model_type == 'uisrcnn':
        config = UISRCNNConfig()
        config.channels = args.channels
        model = UISRCNN_Model(config)
    elif args.model_type == 'uisrcnn_kso':
        config = UISRCNN_kso_Config()
        config.channels = args.channels
        model = UISRCNN_kso_Model(config)
    else:
        raise ValueError('Model type must be either "birnn", "seq2seq", '
                         '"uisrcnn", "uisrcnn_kso" or "srcnn".'
                         )
    return config, model


def run(config, model):
    if args.mode == 'train':
        init = tf.global_variables_initializer()
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        sess = tf.Session()
        sess.run(init)
        model.fit(sess, model.train_data[0], model.train_data[1])

    elif args.mode == 'test':
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        low_resolution = model.test_data[0]
        ground_truths = model.test_data[1]
        prediction = model.predict(sess, low_resolution, ground_truths)

        prediction = prediction.reshape((-1, 1524))
        truth = ground_truths.reshape((-1, 1524))
        low_resolution = low_resolution.reshape((-1, 1524))

        MSE = np.mean((prediction - truth) ** 2)
        ER = np.mean(np.abs(prediction - truth)) / np.mean(np.abs(truth)) * 100

        indices = range(low_resolution.shape[0])
        high_resolution_prediction = np.insert(prediction, indices, low_resolution, axis=0).reshape((-1, 128, 1524))
        high_resolution_ground_truths = np.insert(truth, indices, low_resolution, axis=0).reshape((-1, 128, 1524))

        # generate .npy and .mat files
        # high_resolution_prediction and high_resolution_ground_truths are numpy array
        # we load the original data PICMUS_EXP_CR_RF.mat, replace the 'rf_channels' data to generate the .mat files
        pred_filename = 'Prediction-{}'.format(config.model_prefix)
        high_resolution_prediction_mat = sio.loadmat('PICMUS_EXP_CR_RF.mat')
        high_resolution_ground_truths_mat = sio.loadmat('PICMUS_EXP_CR_RF.mat')
        high_resolution_prediction_mat['rf_channels'] = high_resolution_prediction.transpose((2, 1, 0))
        high_resolution_ground_truths_mat['rf_channels'] = high_resolution_ground_truths.transpose((2, 1, 0))
        np.save('tests/{}.npy'.format(pred_filename), high_resolution_prediction)
        sio.savemat('tests/{}.mat'.format(pred_filename), high_resolution_prediction_mat)
        np.save('tests/Truths.npy', high_resolution_ground_truths)
        sio.savemat('tests/Truths.mat', high_resolution_ground_truths_mat)

        print('MSE: {}, ER: {} %'.format(MSE, ER))

        """
        truth = np.load('tests/Truths.npy')
        pred = np.load('tests/{}.npy'.format(pred_filename))
        plt.plot(truth[0][125][:500], color='red', linewidth=0.5, linestyle='-', label='truth')
        plt.plot(pred[0][125][:500], color='blue', linewidth=0.5, linestyle='-', label='prediction')
        plt.legend(loc='upper left')
        plt.title(pred_filename, fontsize='large', fontweight='bold')
        plt.savefig('tests/{}.png'.format(pred_filename), dpi=300)
        
        """


if __name__ == '__main__':
    config, model = create_model()
    run(config, model)
