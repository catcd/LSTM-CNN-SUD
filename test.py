import pickle

import data.constants as constants
from utils import get_trimmed_w2v_vectors, load_vocab
from model import LstmCnnModel
from sklearn.metrics import precision_recall_fscore_support
from dataset import Dataset


def main():
    vocab_words = load_vocab(constants.ALL_WORDS)
    train = Dataset(constants.RAW_DATA + 'train', vocab_words)
    validation = Dataset(constants.RAW_DATA + 'dev', vocab_words)
    test = Dataset(constants.RAW_DATA + 'test', vocab_words)

    # get pre trained embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_FASTTEXT_W2V)

    model = LstmCnnModel(
        model_name=constants.MODEL_NAMES.format('sud', constants.JOB_IDENTITY),
        embeddings=embeddings,
        batch_size=constants.BATCH_SIZE,
        constants=constants,
    )

    # train, evaluate and interact
    model.build()
    model.load_data(train=train, validation=validation)
    model.run_train(epochs=constants.EPOCHS, early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

    y_pred = model.predict(test)
    preds = []
    labels = []
    for pred, label in zip(y_pred, test.labels):
        labels.extend(label)
        preds.extend(pred[:len(label)])

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print('Result:\tP={:.2f}%\tR={:.2f}%\tF1={:.2f}%'.format(p * 100, r * 100, f1 * 100))


if __name__ == '__main__':
    main()
