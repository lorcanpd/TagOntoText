
import numpy as np
import os
from pathlib import Path
import tensorflow_addons as tfa
from .split_train_data import buffer_count
from .custom_metrics import MultiClassTruePositives, MultiClassFalsePositives, \
    MultiClassFalseNegatives, precision_fn, recall_fn, f1_fn
from six.moves import reduce
import tensorflow as tf
from .stream_dataset import ftags
from tensorflow_addons.text import crf_log_likelihood


class CustomModelBase(tf.keras.Model):

    def __init__(self, params):
        super(CustomModelBase, self).__init__()
        self.params = params

        # Create hashed lookups for vocabulary files.
        self._build_char_hash()
        self._build_word_hash()

        # Create character and word embedding matrices.
        with Path(self.params['tags']).open() as f:
            self.indices = [
                idx for idx, tag in enumerate(f) if tag.strip() != 'O'
            ]
            self.num_tags = len(self.indices) + 1
        with Path(self.params['chars']).open() as f:
            self.num_chars = sum(1 for _ in f) + self.params['num_oov_buckets']
        self._build_char_emb()
        self._build_word_emb()

        # Initialise CRF transition parameter matrix.
        self._init_crf_params()

        self._build_tag_hash_tables()

        # Define other model layers.
        self.char_conv_layer = tf.keras.layers.Conv1D(
            filters=self.params['filter'],
            kernel_size=self.params['kernel_size'],
            padding='same'
        )

        self.bi_lstm_layer = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(
                units=self.params['lstm_size'],
                return_sequences=True
            ),
            merge_mode='concat'
        )

        self.dense_layer = tf.keras.layers.Dense(self.num_tags)

        # Set optimiser.
        self.optimizer = tf.keras.optimizers.Adam()

        self._initialise_metrics()

        try:
            os.makedirs(self.params['checkpoint_dir'], exist_ok=True)
        except FileExistsError:
            pass

    def call(self, data, training=None, **kwargs):

        if isinstance(data, dict):
            data = ((data['words'], data['nwords']),
                    (data['chars'], data['nchars']))
        (words, nwords), (chars, nchars) = data

        char_ids = self.vocab_chars.lookup(chars)
        char_embeddings = tf.nn.embedding_lookup(params=self.char_emb,
                                                 ids=char_ids)

        if training:
            char_embeddings = tf.nn.dropout(x=char_embeddings,
                                            rate=self.params['dropout'])

        # Character 1D convolution.
        weights = tf.sequence_mask(nchars)
        char_embeddings = self.masked_conv1d_and_max(
            char_embeddings,
            weights,
            self.params['filter']
        )

        word_ids = self.vocab_words.lookup(words)
        word_embeddings = tf.nn.embedding_lookup(
            params=self.word_emb, ids=word_ids
        )

        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

        if training:
            embeddings = tf.nn.dropout(x=embeddings,
                                       rate=self.params['dropout'])

        output = self.bi_lstm_layer(embeddings)

        if not training:
            batch_lids = self._lid(logits=output, k=5)
        else:
            batch_lids = None

        if training:
            output = tf.nn.dropout(x=output,
                                   rate=self.params['dropout'])

        logits = self.dense_layer(output)

        pred_ids, _ = tfa.text.crf.crf_decode(potentials=logits,
                                              transition_params=self.crf_params,
                                              sequence_length=nwords)

        return logits, pred_ids, batch_lids

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            logits, pred_ids, _ = self.call(features, training=True)
            loss_value = self._loss_fn(logits,
                                       tags,
                                       features['nwords'],
                                       self.crf_params)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        weights = tf.cast(tf.sequence_mask(features['nwords']), tf.int32)

        self.epoch_train_loss.update_state(loss_value)
        self.epoch_train_accuracy.update_state(tags, pred_ids)
        self.epoch_train_true_positives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_train_false_positives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_train_false_negatives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )

    def test_step(self, features, tags):
        logits, pred_ids, batch_lids = self.call(features, training=False)
        loss_value = self._loss_fn(logits,
                                   tags,
                                   features['nwords'],
                                   self.crf_params)

        weights = tf.cast(tf.sequence_mask(features['nwords']), tf.int32)

        self.epoch_test_loss.update_state(loss_value)
        self.epoch_test_accuracy.update_state(tags, pred_ids)
        self.epoch_test_true_positives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_test_false_positives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_test_false_negatives.update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_test_lid.update_state(batch_lids)

    def _initialise_metrics(self):
        # Training.
        self.train_loss = []
        self.train_accuracy = []
        self.train_precision = []
        self.train_recall = []
        self.train_f1 = []

        self.epoch_train_loss = tf.keras.metrics.Mean(name="loss")
        self.epoch_train_accuracy = tf.keras.metrics.Accuracy(name="accuracy")
        self.epoch_train_true_positives = MultiClassTruePositives()
        self.epoch_train_false_positives = MultiClassFalsePositives()
        self.epoch_train_false_negatives = MultiClassFalseNegatives()

        # Testing.
        self.test_loss = []
        self.test_accuracy = []
        self.test_precision = []
        self.test_recall = []
        self.test_f1 = []
        self.test_lid = []

        self.epoch_test_loss = tf.keras.metrics.Mean(name="loss")
        self.epoch_test_accuracy = tf.keras.metrics.Accuracy(name="accuracy")
        self.epoch_test_true_positives = MultiClassTruePositives()
        self.epoch_test_false_positives = MultiClassFalsePositives()
        self.epoch_test_false_negatives = MultiClassFalseNegatives()
        self.epoch_test_lid = tf.keras.metrics.Mean(name="lid")

        self.current_train_loss = []
        self.current_train_accuracy = []
        self.current_train_precision = []
        self.current_train_recall = []
        self.current_train_f1 = []

    def end_of_epoch_metrics(self, end=True):
        if end:
            self.train_loss.append(self.epoch_train_loss.result())
            self.train_accuracy.append(self.epoch_train_accuracy.result())
            self.train_precision.append(
                precision_fn(self.epoch_train_true_positives.result(),
                             self.epoch_train_false_positives.result())
            )
            self.train_recall.append(
                recall_fn(self.epoch_train_true_positives.result(),
                          self.epoch_train_false_negatives.result())
            )
            self.train_f1.append(
                f1_fn(self.epoch_train_true_positives.result(),
                      self.epoch_train_false_positives.result(),
                      self.epoch_train_false_negatives.result())
            )

            self.epoch_train_loss.reset_states()
            self.epoch_train_accuracy.reset_states()
            self.epoch_train_true_positives.reset_states()
            self.epoch_train_false_positives.reset_states()
            self.epoch_train_false_negatives.reset_states()

            self.test_loss.append(self.epoch_test_loss.result())
            self.test_accuracy.append(self.epoch_test_accuracy.result())
            self.test_precision.append(
                precision_fn(self.epoch_test_true_positives.result(),
                             self.epoch_test_false_positives.result())
            )
            self.test_recall.append(
                recall_fn(self.epoch_test_true_positives.result(),
                          self.epoch_test_false_negatives.result())
            )
            self.test_f1.append(
                f1_fn(self.epoch_test_true_positives.result(),
                      self.epoch_test_false_positives.result(),
                      self.epoch_test_false_negatives.result())
            )
            self.test_lid.append(self.epoch_test_lid.result())

            self.epoch_test_loss.reset_states()
            self.epoch_test_accuracy.reset_states()
            self.epoch_test_true_positives.reset_states()
            self.epoch_test_false_positives.reset_states()
            self.epoch_test_false_negatives.reset_states()
            self.epoch_test_lid.reset_states()

            self.current_train_loss = []
            self.current_train_accuracy = []
            self.current_train_precision = []
            self.current_train_recall = []
            self.current_train_f1 = []
        else:
            self.current_train_loss.append(self.epoch_train_loss.result())
            self.current_train_accuracy.append(
                self.epoch_train_accuracy.result()
            )
            self.current_train_precision.append(
                precision_fn(self.epoch_train_true_positives.result(),
                             self.epoch_train_false_positives.result())
            )
            self.current_train_recall.append(
                recall_fn(self.epoch_train_true_positives.result(),
                          self.epoch_train_false_negatives.result())
            )
            self.current_train_f1.append(
                f1_fn(self.epoch_train_true_positives.result(),
                      self.epoch_train_false_positives.result(),
                      self.epoch_train_false_negatives.result())
            )

    def _test(self, test_data, steps=None):
        for test_batch, features in enumerate(test_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.test_step(features, tags)

            if test_batch == steps:
                break

    def _print_metrics(self, epoch, batch, training=None):
        if training:
            print(
                "Training Epoch {:03d}, Batch {:03d}: Loss: {:.3f}, Accuracy: {:.1%}, Precision: {:.1%}, Recall: {:.1%}, F1: {:.1%}".format(
                    epoch,
                    batch,
                    self.current_train_loss[-1],
                    self.current_train_accuracy[-1],
                    self.current_train_precision[-1],
                    self.current_train_recall[-1],
                    self.current_train_f1[-1]
                )
            )
        else:
            print("\nEnd of epoch {:03d} metrics".format(epoch))
            print(
                "Training Loss: {:.3f}, Accuracy: {:.1%}, Precision: {:.1%}, Recall: {:.1%}, F1: {:.1%}".format(
                    self.train_loss[-1],
                    self.train_accuracy[-1],
                    self.train_precision[-1],
                    self.train_recall[-1],
                    self.train_f1[-1]
                )
            )
            print(
                "Testing Loss: {:.3f}, Accuracy: {:.1%}, Precision: {:.1%}, Recall: {:.1%}, F1: {:.1%}, LID: {:.0f} \n".format(
                    self.test_loss[-1],
                    self.test_accuracy[-1],
                    self.test_precision[-1],
                    self.test_recall[-1],
                    self.test_f1[-1],
                    self.test_lid[-1]
                )
            )

    def _single_epoch(self, train_data, test_data, test_every_n):
        test_num_lines = buffer_count(ftags(self.params['datadir'], 'test'))
        five_perc = int(
            np.floor((test_num_lines / self.params['batch_size']) / 20))
        # num_lines = buffer_count(ftags(self.params['datadir'], 'train'))
        # T = int(np.floor(num_lines/self.params['batch_size']))

        for train_batch, features in enumerate(train_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.train_step(features, tags)

            if train_batch % test_every_n == 0:
                self._test(test_data, five_perc)
                self.end_of_epoch_metrics(end=True)
                self._print_metrics(0, train_batch, training=False)
                self.save_weights(self.params['checkpoint_dir'] +
                                  f"/{self.params['name']}_ckpt.tf")

        self._test(test_data)
        self.end_of_epoch_metrics(end=True)
        print("\nFINAL METRICS")
        self._print_metrics(0, train_batch, training=False)
        self.save_weights(self.params['checkpoint_dir'] +
                          f"/{self.params['name']}_ckpt.tf")

    def _multi_epoch(self, train_data, test_data, update_metrics_every_n):
        num_lines = buffer_count(ftags(self.params['datadir'], 'train'))
        T = int(
            np.floor(
                num_lines / self.params['batch_size']
            )
        )

        epoch = 1
        batch = 1
        for i, features in enumerate(train_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.train_step(features, tags)

            if batch % update_metrics_every_n == 0:
                self.end_of_epoch_metrics(end=False)
                self._print_metrics(epoch, batch, training=True)

            if batch % T == 0 and batch != 0:
                self._test(test_data)
                self.end_of_epoch_metrics(end=True)
                self._print_metrics(epoch, batch, training=False)
                self.save_weights(self.params['checkpoint_dir'] +
                                  f"/{self.params['name']}_ckpt.tf")

                epoch += 1
                batch = 1
                if epoch > self.params['epochs']:
                    break
            else:
                batch += 1
        self.save_weights(self.params['checkpoint_dir'] +
                          f"/{self.params['name']}_ckpt.tf")

    def train_and_eval(self, train_data, test_data):
        if self.params['epochs'] == 1:
            self._single_epoch(train_data, test_data, test_every_n=50)
        elif self.params['epochs'] > 1:
            self._multi_epoch(train_data, test_data, update_metrics_every_n=100)

    def _build_char_hash(self):
        self.vocab_chars = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=self.params['chars'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\n"
            ),
            num_oov_buckets=self.params['num_oov_buckets'],
            lookup_key_dtype=tf.string
        )

    def _build_word_hash(self):
        self.vocab_words = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=self.params['words'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\n"
            ),
            num_oov_buckets=self.params['num_oov_buckets']
        )

    def _build_word_emb(self):
        glove = np.load(self.params['glove'])['embeddings']
        variable = np.vstack([glove, [[0.] * self.params['dim']]])
        self.word_emb = tf.Variable(
            variable,
            shape=variable.shape,
            dtype=tf.float32,
            trainable=False
        )

    def _build_char_emb(self):
        self.char_emb = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[self.num_chars + 1, self.params['dim_chars']],
                minval=-1,
                maxval=1,
                dtype=tf.float32
            ),
            name='chars_embedding',
            shape=[self.num_chars + 1, self.params['dim_chars']],
            dtype=tf.float32
        )

    def _init_crf_params(self):
        self.crf_params = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[self.num_tags, self.num_tags],
                minval=0,
                maxval=1,
                dtype=tf.float32
            ),
            name="crf_transition_parameters",
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32
        )

    def _build_tag_hash_tables(self):
        # Tag to index.
        self.vocab_tags = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=self.params['tags'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\n"
            ),
            num_oov_buckets=1,
            lookup_key_dtype=tf.string
        )
        # Index to tag.
        self.reverse_vocab_tags = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename=self.params['tags'],
                key_dtype=tf.int64,
                key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                value_dtype=tf.string,
                value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                delimiter="\n"
            ),
            default_value='O'
        )

    def masked_conv1d_and_max(self, t, weights, filters):
        """Applies 1d convolution and a masked max-pooling
        Parameters
        ----------
        t : tf.Tensor
            A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
        weights : tf.Tensor of tf.bool
            A Tensor of shape [d1, d2, dn-1]
        filters : int
            number of filters
        # kernel_size : int
        #     kernel size for the temporal convolution
        Returns
        -------
        tf.Tensor
            A tensor of shape [d1, d2, dn-1, filters]
        """
        # Get shape and parameters
        shape = tf.shape(t)
        ndims = t.shape.ndims
        dim1 = reduce(lambda x, y: x * y,
                      [shape[i] for i in range(ndims - 2)])
        dim2 = shape[-2]
        dim3 = t.shape[-1]
        # Reshape weights
        weights = tf.reshape(weights, shape=[dim1, dim2, 1])
        weights = tf.cast(weights, dtype=tf.float32)
        # Reshape input and apply weights
        flat_shape = [dim1, dim2, dim3]
        t = tf.reshape(t, shape=flat_shape)
        t *= weights
        # Apply convolution
        t_conv = self.char_conv_layer(t)
        t_conv *= weights
        # Reduce max -- set to zero if all padded
        t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2,
                                                 keepdims=True)
        t_max = tf.reduce_max(t_conv, axis=-2)
        # Reshape the output
        final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
        t_max = tf.reshape(t_max, shape=final_shape)

        return t_max

    def _loss_fn(self, logits, tags, nwords, crf_params):
        log_likelihood, _ = crf_log_likelihood(
            inputs=logits,
            tag_indices=tags,
            sequence_lengths=nwords,
            transition_params=crf_params
        )
        loss = tf.compat.v1.reduce_mean(-log_likelihood)

        return loss

    @staticmethod
    def _lid(logits, k=20):
        """
        Calculate LID for each data point in the array.
        :param logits:
        :param k:
        :return:
        """

        if len(tf.shape(logits)) == 3:
            logits = tf.reshape(logits, [-1, tf.shape(logits)[2]])

        batch_size = tf.shape(logits)[0]
        # n_samples = logits.get_shape().as_list()
        # calculate pairwise distance
        r = tf.reduce_sum(logits * logits, 1)
        # turn r into column vector
        r1 = tf.reshape(r, [-1, 1])
        d = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
            tf.ones([batch_size, batch_size])

        # find the k nearest neighbor
        d1 = -tf.sqrt(d)
        d2, _ = tf.nn.top_k(d1, k=k, sorted=True)
        d3 = -d2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

        m = tf.transpose(tf.multiply(tf.transpose(d3), 1.0 / d3[:, -1]))
        v_log = tf.reduce_sum(tf.math.log(m + tf.keras.backend.epsilon()), axis=1)  # to avoid nan
        lids = -k / v_log

        return lids
