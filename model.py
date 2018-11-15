import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.rnn import BasicLSTMCell

from dataset import pad_sequences
from utils import Timer, Log

seed = 13
np.random.seed(seed)


class LstmCnnModel:
    def __init__(self, model_name, embeddings, batch_size, constants):
        self.model_name = model_name
        self.embeddings = embeddings
        self.batch_size = batch_size

        self.use_lstm = constants.USE_LSTM
        self.output_lstm_dims = constants.OUTPUT_LSTM_DIMS

        self.use_cnn = constants.USE_CNN
        self.cnn_filters = constants.CNN_FILTERS

        if not self.use_cnn and not self.use_lstm:
            raise ValueError('Config at least 1 channel LSTM or CNN and start again!')

        self.hidden_layers = constants.HIDDEN_LAYERS

        self.all_labels = constants.ALL_LABELS
        self.num_of_class = len(constants.ALL_LABELS)

        self.trained_models = constants.TRAINED_MODELS

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.word_ids = tf.placeholder(name='word_ids', shape=[None, None], dtype='int32')
        self.labels = tf.placeholder(name='y_true', shape=[None, None], dtype='int32')
        self.sequence_lens = tf.placeholder(name='sequence_lens', dtype=tf.int32, shape=[None])
        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_embedding')
        self.dropout_lstm = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_lstm')
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.dropout_hidden = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_hidden')
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope('word_embedding'):
            _embeddings_lut = tf.Variable(self.embeddings, name='lut', dtype=tf.float32, trainable=False)
            self.word_embeddings = tf.nn.embedding_lookup(
                _embeddings_lut, self.word_ids,
                name='embeddings'
            )
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_embedding)

    @staticmethod
    def batch_normalization(inputs, training, decay=0.9, epsilon=1e-3):
        scale = tf.get_variable('scale', inputs.get_shape()[-1], initializer=tf.ones_initializer(),
                                dtype=tf.float32)
        beta = tf.get_variable('beta', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                               dtype=tf.float32)
        pop_mean = tf.get_variable('pop_mean', inputs.get_shape()[-1], initializer=tf.zeros_initializer(),
                                   dtype=tf.float32, trainable=False)
        pop_var = tf.get_variable('pop_var', inputs.get_shape()[-1], initializer=tf.ones_initializer(),
                                  dtype=tf.float32, trainable=False)

        axis = list(range(len(inputs.get_shape()) - 1))

        def Train():
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            pop_mean_new = pop_mean * decay + batch_mean * (1 - decay)
            pop_var_new = pop_var * decay + batch_var * (1 - decay)
            with tf.control_dependencies([pop_mean.assign(pop_mean_new), pop_var.assign(pop_var_new)]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

        def Eval():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(training, Train, Eval)

    def _add_logits_op(self):
        """
        Adds logits to self
        """

        with tf.variable_scope('bi_lstm_fasttext'):
            if self.use_lstm:
                cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                    [BasicLSTMCell(size) for size in self.output_lstm_dims]
                )
                cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                    [BasicLSTMCell(size) for size in self.output_lstm_dims]
                )

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    self.word_embeddings,
                    sequence_length=self.sequence_lens,
                    dtype=tf.float32,
                )
                with tf.variable_scope('bi_lstm_f'):
                    output_fw = self.batch_normalization(output_fw, training=self.is_training)
                with tf.variable_scope('bi_lstm_b'):
                    output_bw = self.batch_normalization(output_bw, training=self.is_training)

                lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                lstm_output = tf.nn.dropout(lstm_output, self.dropout_lstm)
            else:
                lstm_output = None

        with tf.variable_scope('cnn'):
            if self.use_cnn:
                cnn_outputs = []
                for k in self.cnn_filters:
                    with tf.variable_scope('cnn-{}'.format(k)):
                        filters = self.cnn_filters[k]
                        cnn_op = tf.layers.conv1d(
                            self.word_embeddings, filters=filters,
                            kernel_size=k,
                            padding='same', name='cnn-{}'.format(k),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                        )
                        cnn_op = self.batch_normalization(cnn_op, training=self.is_training)
                        cnn_outputs.append(cnn_op)

                cnn_output = tf.concat(cnn_outputs, axis=-1)
                cnn_output = tf.nn.dropout(cnn_output, self.dropout_cnn)
            else:
                cnn_output = None

        with tf.variable_scope('proj'):
            if self.use_lstm:
                nsteps = tf.shape(lstm_output)[1]
            else:
                nsteps = tf.shape(cnn_output)[1]

            if self.use_lstm:
                with tf.variable_scope('lstm_logit'):
                    feature_dim = 2 * self.output_lstm_dims[-1]
                    output = tf.reshape(lstm_output, [-1, feature_dim])

                    for i, v in enumerate(self.hidden_layers, start=1):
                        output = tf.layers.dense(
                            inputs=output, units=v, name='hidden_{}'.format(i),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            activation=tf.nn.tanh,
                        )
                        output = tf.nn.dropout(output, self.dropout_hidden)

                    output = tf.layers.dense(
                        inputs=output, units=2, name='final_dense',
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    self.lstm_logits = tf.reshape(output, [-1, nsteps, 2])
            else:
                self.lstm_logits = None

            if self.use_cnn:
                with tf.variable_scope('cnn_logit'):
                    feature_dim = sum(self.cnn_filters[k] for k in self.cnn_filters)
                    output = tf.reshape(cnn_output, [-1, feature_dim])

                    for i, v in enumerate(self.hidden_layers, start=1):
                        output = tf.layers.dense(
                            inputs=output, units=v, name='hidden_{}'.format(i),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            activation=tf.nn.tanh,
                        )
                        output = tf.nn.dropout(output, self.dropout_hidden)

                    output = tf.layers.dense(
                        inputs=output, units=2, name='final_dense',
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                    )
                    self.cnn_logits = tf.reshape(output, [-1, nsteps, 2])
            else:
                self.cnn_logits = None

            with tf.variable_scope('combined_logit'):
                outputs = []
                if self.use_lstm:
                    outputs.append(lstm_output)
                if self.use_cnn:
                    outputs.append(cnn_output)
                combined_output = tf.concat(outputs, axis=-1)

                feature_dim = (
                    (2 * self.output_lstm_dims[-1] if len(self.output_lstm_dims) != 0 else 0)
                    + sum(self.cnn_filters[k] for k in self.cnn_filters)
                )
                output = tf.reshape(combined_output, [-1, feature_dim])

                for i, v in enumerate(self.hidden_layers, start=1):
                    output = tf.layers.dense(
                        inputs=output, units=v, name='hidden_{}'.format(i),
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=tf.nn.tanh,
                    )
                    output = tf.nn.dropout(output, self.dropout_hidden)

                output = tf.layers.dense(
                    inputs=output, units=self.num_of_class, name='final_dense',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                )
                self.combined_logits = tf.reshape(output, [-1, nsteps, self.num_of_class])
                combined_logits = self.batch_normalization(self.combined_logits, training=self.is_training)

        with tf.variable_scope('final_logit'):
            self.logits = tf.nn.softmax(combined_logits)

            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            mask = tf.sequence_mask(self.sequence_lens)
            self.loss = 0

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.combined_logits, labels=self.labels)
            losses = tf.boolean_mask(losses, mask)
            self.loss += tf.reduce_mean(losses)

            label_binary = tf.minimum(self.labels, 1)
            if self.use_lstm:
                lstm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.lstm_logits, labels=label_binary)
                lstm_losses = tf.boolean_mask(lstm_losses, mask)
                self.loss += tf.reduce_mean(lstm_losses)

            if self.use_cnn:
                cnn_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cnn_logits, labels=label_binary)
                cnn_losses = tf.boolean_mask(cnn_losses, mask)
                self.loss += tf.reduce_mean(cnn_losses)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('train_step'):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        timer = Timer()
        timer.start('Building model...')

        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_logits_op()
        self._add_loss_op()
        self._add_train_op()

        timer.stop()

    def _loss(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_lstm] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden] = 1.0
        feed_dict[self.is_training] = False

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            w_batch = data['words'][start:start + self.batch_size]
            l_batch = data['labels'][start:start + self.batch_size]

            word_ids, sequence_lengths = pad_sequences(w_batch, pad_tok=0)
            labels, _ = pad_sequences(l_batch, pad_tok=0)

            start += self.batch_size
            idx += 1
            yield (word_ids, labels, sequence_lengths)

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=2)
        best_loss = float('inf')
        nepoch_noimp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch_train = int(np.ceil(len(self.dataset_train.words)/self.batch_size))

            for e in range(epochs):
                words_shuffled, labels_shuffled = shuffle(
                    self.dataset_train.words,
                    self.dataset_train.labels,
                )

                data = {
                    'words': words_shuffled,
                    'labels': labels_shuffled,
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                    words, labels, sequence_lengths = batch
                    feed_dict = {
                        self.word_ids: words,
                        self.labels: labels,
                        self.sequence_lens: sequence_lengths,
                        self.dropout_embedding: 0.5,
                        self.dropout_lstm: 0.5,
                        self.dropout_cnn: 0.5,
                        self.dropout_hidden: 0.5,
                        self.is_training: True,
                    }

                    _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        Log.log('Iter {}, Loss: {} '.format(idx, loss_train))

                Log.log('End epochs {}'.format(e + 1))

                # early stop by validation loss
                if early_stopping:
                    num_batch_val = len(self.dataset_validation.words) // self.batch_size + 1
                    total_loss = []

                    data = {
                        'words': self.dataset_validation.words,
                        'labels': self.dataset_validation.labels,
                    }

                    for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                        words, labels, sequence_lengths = batch

                        loss = self._loss(sess, feed_dict={
                            self.word_ids: words,
                            self.labels: labels,
                            self.sequence_lens: sequence_lengths,
                        })

                        total_loss.append(loss)

                    val_loss = np.mean(total_loss)
                    Log.log('Val loss: {}'.format(val_loss))
                    if val_loss < best_loss:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_loss = val_loss
                        nepoch_noimp = 0
                    else:
                        nepoch_noimp += 1
                        Log.log('Number of epochs with no improvement: {}'.format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            Log.log('Best loss: {}'.format(best_loss))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def load_data(self, train, validation):
        """
        :param dataset.Dataset train:
        :param dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print('Number of training examples:', len(self.dataset_train.labels))
        print('Number of validation examples:', len(self.dataset_validation.labels))
        timer.stop()

    def run_train(self, epochs, early_stopping=True, patience=10):
        timer = Timer()
        timer.start('Training model...')
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
        timer.stop()

    # test
    def predict(self, test):
        """

        :param dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log('Testing model over test set')
            saver.restore(sess, self.model_name)

            y_pred = []
            num_batch = len(test.labels) // self.batch_size + 1

            data = {
                'words': test.words,
                'labels': test.labels,
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
                words, labels, sequence_lengths = batch
                feed_dict = {
                    self.word_ids: words,
                    self.sequence_lens: sequence_lengths,
                    self.dropout_embedding: 1.0,
                    self.dropout_lstm: 1.0,
                    self.dropout_cnn: 1.0,
                    self.dropout_hidden: 1.0,
                    self.is_training: False,
                }
                preds = sess.run(self.labels_pred, feed_dict=feed_dict)
                y_pred.extend(preds)

        return y_pred
