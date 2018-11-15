import argparse

ALL_LABELS = ['SU', 'NonSU']

parser = argparse.ArgumentParser(description='Hybrid biLSTM and CNN architecture for Sentence Unit Detection')

parser.add_argument('-i', help='Job identity', type=int, default=0)

parser.add_argument('-e', help='Number of epochs', type=int, default=20)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=0)
parser.add_argument('-b', help='Batch size', type=int, default=8)

parser.add_argument('-lstm', help='Number of output fastText w2v embedding LSTM dimension', type=str, default='100,100')
parser.add_argument('-cnn', help='CNN configurations', type=str, default='3:32,5:64,7:32')

parser.add_argument('-hd', help='Hidden layer configurations', type=str, default='128,128,128')

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i

EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p
BATCH_SIZE = opt.b

INPUT_EMBEDDING_DIM = 300

USE_LSTM = False if opt.lstm == '0' else True
OUTPUT_LSTM_DIMS = list(map(int, opt.lstm.split(','))) if opt.lstm != '0' else []

USE_CNN = False if opt.cnn == '0' else True
CNN_FILTERS = {
    int(k): int(f) for k, f in [i.split(':') for i in opt.cnn.split(',')]
} if opt.cnn != '0' else {}

HIDDEN_LAYERS = list(map(int, opt.hd.split(','))) if opt.hd != '0' else []

DATA = 'data/'
RAW_DATA = DATA + 'raw_data/'
PARSED_DATA = DATA + 'parsed_data/'
PICKLE_DATA = DATA + 'pickle/'

ALL_WORDS = PARSED_DATA + 'all_words.txt'

W2V_DATA = DATA + 'w2v_model/'
TRIMMED_FASTTEXT_W2V = W2V_DATA + 'trimmed_w2v.npz'

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
