import numpy as np
from collections import Counter
import itertools

np.random.seed(13)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


class Dataset:
    def __init__(self, data_name, vocab_words=None, init=True):
        self.data_name = data_name

        self.words = None
        self.labels = None

        self.vocab_words = vocab_words

        if init:
            self._process_data()
            self._clean_data()

    def _clean_data(self):
        del self.vocab_words

    def _process_data(self):
        data_words, labels = self._parse_raw()
        words = []

        for i in data_words:
            ws = [self.vocab_words[w] if w in self.vocab_words else self.vocab_words['$UNK$'] for w in i]
            words.append(ws)

        self.words = words
        self.labels = labels

    def _parse_raw(self):
        all_words = []
        all_labels = []

        with open('{}.word.txt'.format(self.data_name), 'r') as f:
            for line in f:
                all_words.append(line.strip().split())

        with open('{}.label.txt'.format(self.data_name), 'r') as f:
            for line in f:
                all_labels.append(list(map(int, line.strip().split())))

        return all_words, all_labels
