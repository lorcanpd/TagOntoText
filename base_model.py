
import numpy as np
import os
from pathlib import Path
import tensorflow_addons as tfa
from .split_train_data import buffer_count
from .custom_metrics import MultiClassTruePositives, MultiClassFalsePositives, \
    MultiClassFalseNegatives, precision_fn, recall_fn, f1_fn
from six.moves import reduce
import tensorflow as tf
# from .stream_dataset import ftags
from tensorflow_addons.text import crf_log_likelihood
from .stream_dataset import fwords, ftags, inputter
import pandas as pd
# from regex import sub

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

        if training:
            # batch_lids = self._lid(logits=output, k=5)
            # not_inf = tf.math.is_finite(batch_lids)
            # batch_lids = tf.boolean_mask(batch_lids, not_inf)
            for_lids = output
        else:
            for_lids = None
            # batch_lids = None

        if training:
            output = tf.nn.dropout(x=output,
                                   rate=self.params['dropout'])

        logits = self.dense_layer(output)

        pred_ids, _ = tfa.text.crf.crf_decode(potentials=logits,
                                              transition_params=self.crf_params,
                                              sequence_length=nwords)

        return logits, pred_ids, for_lids # batch_lids

    def predict_step(self, features):
        _, pred_ids, _ = self.call(features, training=False)
        pred_tags = self.reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        return pred_tags

    # TODO: This function produces nonsense tags. Need to figure out why but
    #  currently it's not a priority.
    # def predict_single(self, sentence):
    #     words = [w.encode() for w in sentence.strip().split()]
    #
    #     if len(words) < 5:
    #         print("Sentence must have at least 5 words.")
    #         return None
    #
    #     chars = [[c.encode() for c in w] for w in sentence.strip().split()]
    #     lengths = [len(c) for c in chars]
    #     max_len = max(lengths)
    #     chars = [c + [b"<pad>"] * (max_len - l) for c, l in zip(chars, lengths)]
    #     words = tf.expand_dims(tf.cast(words, tf.string), axis=0)
    #     n_words = tf.expand_dims(tf.cast(len(words), tf.int32), axis=0)
    #     chars = tf.expand_dims(tf.cast(chars, tf.string), axis=0)
    #     lengths = tf.expand_dims(tf.cast(lengths, tf.int32), axis=0)
    #     data = (words, n_words), (chars, lengths)
    #
    #     pred_tags = self.predict_step(data)
    #
    #     return pred_tags

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            # logits, pred_ids, batch_lids = self.call(features, training=True)
            logits, pred_ids, for_lids = self.call(features, training=True)
            loss_value = self._loss_fn(logits,
                                       tags,
                                       features['nwords'],
                                       self.crf_params)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        mask = tf.sequence_mask(features['nwords'])

        # Calculate latent intrinsic dimensionality.
        for_lids = tf.boolean_mask(for_lids, mask)
        batch_lids = self._lid(for_lids, k=5)

        weights = tf.cast(mask, tf.int32)

        self.epoch_train_metrics['loss'].update_state(loss_value)
        self.epoch_train_metrics['accuracy'].update_state(tags, pred_ids)
        self.epoch_train_metrics['true_positives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_train_metrics['false_positives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_train_metrics['false_negatives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_train_metrics['lid'].update_state(batch_lids)
        # LID RECORDING LIST
        # self.lid_utils['lid'].append(tf.reduce_mean(batch_lids))
        # self.lid_utils['policy_factor'].append(self.policy)

    def test_step(self, features, tags):
        logits, pred_ids, _ = self.call(features, training=False)
        loss_value = self._loss_fn(logits,
                                   tags,
                                   features['nwords'],
                                   self.crf_params)

        weights = tf.cast(tf.sequence_mask(features['nwords']), tf.int32)

        self.epoch_test_metrics['loss'].update_state(loss_value)
        self.epoch_test_metrics['accuracy'].update_state(tags, pred_ids)
        self.epoch_test_metrics['true_positives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_test_metrics['false_positives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )
        self.epoch_test_metrics['false_negatives'].update_state(
            tags, pred_ids, self.num_tags, self.indices, weights
        )

    def _initialise_metrics(self):
        # Training.
        self.train_metrics = {}
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1',
                  'lid',
                  'iter']:
            self.train_metrics[m] = []
        # self.lid_utils = {
        #     'lid': []#,
        #     # 'iter': [],
        #     # 'policy_factor': []
        # }

        self.epoch_train_metrics = {
            'loss': tf.keras.metrics.Mean(name="loss"),
            'accuracy': tf.keras.metrics.Accuracy(name="accuracy"),
            'true_positives': MultiClassTruePositives(),
            'false_positives': MultiClassFalsePositives(),
            'false_negatives': MultiClassFalseNegatives(),
            # 'lid': []
            'lid': tf.keras.metrics.Mean(name="lid")
        }

        # Testing.
        self.test_metrics = {}
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            self.test_metrics[m] = []
        self.epoch_test_metrics = {
            'loss': tf.keras.metrics.Mean(name="loss"),
            'accuracy': tf.keras.metrics.Accuracy(name="accuracy"),
            'true_positives': MultiClassTruePositives(),
            'false_positives': MultiClassFalsePositives(),
            'false_negatives': MultiClassFalseNegatives()
            # 'lid': tf.keras.metrics.Mean(name="lid")
        }

        self.current_train_metrics = {}
        for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'lid']:
            self.current_train_metrics[m] = []

    def end_of_epoch_metrics(self, i, end=True):
        if end:
            for m in ['loss', 'accuracy']:
                self.train_metrics[m].append(
                    self.epoch_train_metrics[m].result()
                )
            self.train_metrics['precision'].append(
                precision_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_positives'].result()
                )
            )
            self.train_metrics['recall'].append(
                recall_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_negatives'].result()
                )
            )
            self.train_metrics['f1'].append(
                f1_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_positives'].result(),
                    self.epoch_train_metrics['false_negatives'].result()
                )
            )
            # if len(self.lid_utils['lid']) < iter_per:
            #     value = tf.reduce_mean(self.lid_utils['lid'])
            # else:
            #     value = tf.reduce_mean(self.lid_utils['lid'][-iter_per:])
            self.train_metrics['lid'].append(
                self.epoch_train_metrics['lid'].result()
            )
            # tf.reduce_mean(self.lid_utils['lid'][-5:])
            self.train_metrics['iter'].append(i)

            for k, v in self.epoch_train_metrics.items():
                # if k is not 'lid':
                v.reset_states()

            for m in ['loss', 'accuracy']:
                self.test_metrics[m].append(
                    self.epoch_test_metrics[m].result()
                )
            self.test_metrics['precision'].append(
                precision_fn(
                    self.epoch_test_metrics['true_positives'].result(),
                    self.epoch_test_metrics['false_positives'].result()
                )
            )
            self.test_metrics['recall'].append(
                recall_fn(
                    self.epoch_test_metrics['true_positives'].result(),
                    self.epoch_test_metrics['false_negatives'].result()
                )
            )
            self.test_metrics['f1'].append(
                f1_fn(
                    self.epoch_test_metrics['true_positives'].result(),
                    self.epoch_test_metrics['false_positives'].result(),
                    self.epoch_test_metrics['false_negatives'].result()
                )
            )

            for _, v in self.epoch_test_metrics.items():
                v.reset_states()

            self.current_train_metrics = {}
            for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'lid']:
                self.current_train_metrics[m] = []

        else:
            for m in ['loss', 'accuracy']:
                self.current_train_metrics[m].append(
                    self.epoch_train_metrics[m].result()
                )
            self.current_train_metrics['precision'].append(
                precision_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_positives'].result()
                )
            )
            self.current_train_metrics['recall'].append(
                recall_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_negatives'].result()
                )
            )
            self.current_train_metrics['f1'].append(
                f1_fn(
                    self.epoch_train_metrics['true_positives'].result(),
                    self.epoch_train_metrics['false_positives'].result(),
                    self.epoch_train_metrics['false_negatives'].result()
                )
            )
            # if len(self.lid_utils['lid']) < iter_per:
            #     value = tf.reduce_mean(self.lid_utils['lid'])
            # else:
            #     value = tf.reduce_mean(self.lid_utils['lid'][-iter_per:])
            self.current_train_metrics['lid'].append(
                self.epoch_train_metrics['lid'].result()
            )

    def _test(self, test_data, steps=None):
        for test_batch, features in enumerate(test_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.test_step(features, tags)

            if test_batch == steps:
                break

    def validation(self, validation_data):
        self._test(validation_data)
        validation_metrics = {}
        for m in ['loss', 'accuracy']:
            validation_metrics[m] = self.epoch_test_metrics[m].result()
        validation_metrics['precision'] = precision_fn(
                self.epoch_test_metrics['true_positives'].result(),
                self.epoch_test_metrics['false_positives'].result()
        )
        validation_metrics['recall'] = recall_fn(
                self.epoch_test_metrics['true_positives'].result(),
                self.epoch_test_metrics['false_negatives'].result()
        )
        validation_metrics['f1'] = f1_fn(
                self.epoch_test_metrics['true_positives'].result(),
                self.epoch_test_metrics['false_positives'].result(),
                self.epoch_test_metrics['false_negatives'].result()
        )
        s =  (
            "Validation Loss: {loss:.3f}, Accuracy: {accuracy:.1%}, "
            "Precision: {precision:.1%}, "
            "Recall: {recall:.1%}, F1: {f1:.1%}"
        ).format(
            **validation_metrics
        )
        print(s)

    def _print_metrics(self, epoch, batch, training=None):
        if training:
            print_dict = {}
            for k, v in self.current_train_metrics.items():
                print_dict[k] = v[-1]
            s = (
                "Training Epoch {:03d}, Batch {:03d}: Loss: {loss:.3f}, "
                "Accuracy: {accuracy:.1%}, Precision: {precision:.1%}, "
                "Recall: {recall:.1%}, F1: {f1:.1%}, Mean LID: {lid:.1f}").format(
                epoch, batch, **print_dict
            )
            print(s)
        else:
            print("\nEnd of epoch {:03d} metrics".format(epoch))

            print_dict = {}
            for k, v in self.train_metrics.items():
                print_dict[k] = v[-1]
            s = (
                "Training Loss: {loss:.3f}, "
                "Accuracy: {accuracy:.1%}, Precision: {precision:.1%}, "
                "Recall: {recall:.1%}, F1: {f1:.1%}, "
                "Mean LID: {lid:.1f}").format(
                epoch, batch, **print_dict
            )
            print(s)
            print_dict = {}
            for k, v in self.test_metrics.items():
                print_dict[k] = v[-1]
            s = (
                "Testing Loss: {loss:.3f}, "
                "Accuracy: {accuracy:.1%}, Precision: {precision:.1%}, "
                "Recall: {recall:.1%}, F1: {f1:.1%} \n").format(
                epoch, batch, **print_dict
            )
            print(s)

    def _single_epoch(self, train_data, test_data, test_every_n):
        test_num_lines = buffer_count(ftags(self.params['datadir'], 'test'))
        five_perc = int(
            np.floor((test_num_lines / self.params['batch_size']) / 20))

        for train_batch, features in enumerate(train_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.train_step(features, tags)

            # if train_batch % 25 == 0:
            #     self.lid_utils['lid'].append(self.epoch_train_metrics['lid'].result())
            #     self.epoch_train_metrics['lid'].reset_states()
            #     self.lid_utils['iter'].append(train_batch)
            #     self.lid_utils['policy_factor'].append(1)

            if train_batch % test_every_n == 0:
                self._test(test_data, five_perc)
                self.end_of_epoch_metrics(i=train_batch, end=True)
                self._print_metrics(0, train_batch, training=False)
                self.save_weights(self.params['checkpoint_dir'] +
                                  f"/{self.params['name']}_ckpt.tf")

        self._test(test_data)
        self.end_of_epoch_metrics(i=train_batch, end=True)
        print("\nFINAL METRICS")
        self._print_metrics(0, train_batch, training=False)
        self.save_weights(self.params['checkpoint_dir'] +
                          f"/{self.params['name']}_ckpt.tf")

    def _multi_epoch(self, train_data, test_data, update_metrics_every_n):
        num_lines = buffer_count(ftags(self.params['datadir'], 'train'))
        # iter_per_epoch = int(np.floor(num_lines / self.params['batch_size']))
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

            # if batch % 25 == 0:
            #     self.lid_utils['lid'].append(self.epoch_train_metrics['lid'].result())
            #     self.epoch_train_metrics['lid'].reset_states()
            #     self.lid_utils['iter'].append(i)
            #     self.lid_utils['policy_factor'].append(1)

            if batch % update_metrics_every_n == 0:
                self.end_of_epoch_metrics(i=i, end=False)
                self._print_metrics(epoch, batch, training=True)


            if batch % T == 0 and batch != 0:
                self._test(test_data)
                self.end_of_epoch_metrics(i=i, end=True)
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

        # if len(tf.shape(logits)) == 3:
        #     logits = tf.reshape(logits, [-1, tf.shape(logits)[2]])

        batch_size = tf.shape(logits)[0]
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

    # @staticmethod
    # def _lid(logits, k=20):
    #     """
    #     Calculate LID for each data point in the array.
    #     :param logits:
    #     :param k:
    #     :return:
    #     """
    #
    #     if len(tf.shape(logits)) == 3:
    #         logits = tf.reshape(logits, [-1, tf.shape(logits)[2]])
    #
    #     batch_size = tf.shape(logits)[0]
    #     # calculate pairwise distance
    #     r = tf.reduce_sum(logits * logits, 1)
    #     # turn r into column vector
    #     r1 = tf.reshape(r, [-1, 1])
    #     d = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
    #         tf.ones([batch_size, batch_size])
    #
    #     # find the k nearest neighbor
    #     d1 = -tf.sqrt(d)
    #     d2, _ = tf.nn.top_k(d1, k=k, sorted=True)
    #     d3 = -d2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]
    #
    #     m = tf.transpose(tf.multiply(tf.transpose(d3), 1.0 / d3[:, -1]))
    #     v_log = tf.reduce_sum(tf.math.log(m + tf.keras.backend.epsilon()), axis=1)  # to avoid nan
    #     lids = -k / v_log
    #
    #     return lids

    def write_predictions(self, filedir, filename):
        Path("Results/score").mkdir(parents=True, exist_ok=True)
        with Path(
                f'Results/score/{self.params["name"]}_{filename}_preds.txt'
        ).open('wb') as file:
            test_inpf = inputter(fwords(filedir, filename),
                                 ftags(filedir, filename),
                                 self.params)
            for inp_data in test_inpf:
                tags = inp_data.pop('tags')
                preds = self.predict_step(inp_data)
                words = inp_data.pop('words')
                for i in range(len(words)):
                    for word, tag, tag_pred in zip(words[i], tags[i], preds[i]):
                        if word.numpy() == b'<pad>':
                            break
                        file.write(b' '.join([word.numpy(), tag.numpy(),
                                              tag_pred.numpy()]) + b'\n')
                    file.write(b'\n')


    def export_metrics(self):
        # breakpoint()
        train_dict = {k: [i if isinstance(i, int) else i.numpy() for i in v]
                      for k, v in self.train_metrics.items()}
        test_dict = {k: [i if isinstance(i, int) else i.numpy() for i in v]
                     for k, v in self.test_metrics.items()}
        train = pd.DataFrame(train_dict).add_prefix('train_')
        test = pd.DataFrame(test_dict).add_prefix('test_')
        pd.concat([train, test], axis=1).to_csv(
            "{}/{}_training_metrics.csv".format(
                self.params['datadir'], self.params['name']
            )
        )
        # lid_dict = {k: [i if isinstance(i, int) else i.numpy() for i in v]
        #             for k, v in self.lid_utils.items()}
        # pd.DataFrame(lid_dict).to_csv(
        #     "{}/{}_lid_recording.csv".format(
        #         self.params['datadir'], self.params['name']
        #     )
        # )


# TODO: fix the functions below (and predict_single) so that an interactive
#  prediction shell can be used once the model is trained.

#     @staticmethod
#     def align_data(data):
#         spacings = [max([len(seq[i]) for seq in data.values()])
#                     for i in range(len(data[list(data.keys())[0]]))]
#         data_aligned = dict()
#
#         # for each entry, create aligned string
#         for key, seq in data.items():
#             str_aligned = ""
#             for token, spacing in zip(seq, spacings):
#                 str_aligned += token + " " * (spacing - len(token) + 1)
#
#             data_aligned[key] = str_aligned
#
#         return data_aligned
#
#     def interactive_shell(self):
#         print(
#             """
# This is an interactive mode.
# To exit, enter 'exit'.
# You can enter a sentence like so:
# input> blah blah blah.
#
#             """
#         )
#         while True:
#             raw_sentence = input("input> ")
#             sentence = sub(r"\s*(\((?>[^()]+|(?1))*\))$", "", raw_sentence)
#
#             if sentence.strip().split() == ["exit"]:
#                 break
#
#             pred_dict = self.predict(sentence)
#             if pred_dict is None:
#                 continue
#             to_print = self.align_data(pred_dict)
#
#             for word, tag in to_print.items():
#                 print(tag, end="\n")
