import numpy as np
import os
from pathlib import Path
import tensorflow_addons as tfa
from .custom_crf import crf_log_likelihood as cust_crf_log_likelihood
from .custom_crf import lid
from .split_train_data import buffer_count
from .custom_metrics import MultiClassTruePositives, MultiClassFalsePositives, \
    MultiClassFalseNegatives, precision_fn, recall_fn, f1_fn
from six.moves import reduce
import tensorflow as tf
from .stream_dataset import ftags

class CustomModelBase(tf.keras.Model):

    def __init__(self, params):
        super(CustomModelBase, self).__init__()
        self.params = params

        if self.name == 'vanilla':
            pass
        elif self.name == 'ddl':
            pass
        elif self.name == 'aere':
            pass
        else:
            raise ValueError(
                f"{self.name} is invalid. Please use 'vanilla', 'ddl', or 'aere'."
            )

        try:
            os.mkdir(self.params['checkpoint_dir'])
        except FileExistsError:
            pass

        self.policy_factor = tf.cast(1, tf.float32)

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
            batch_lids = lid(logits=output, k=5)
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
            loss_value = loss_fn(logits,
                                 tags,
                                 features['nwords'],
                                 self.crf_params,
                                 self.policy_factor)
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
        loss_value = loss_fn(logits,
                             tags,
                             features['nwords'],
                             self.crf_params,
                             self.policy_factor)

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


    def train_and_eval(self, train_data, test_data):

        def _test(steps=None):
            for test_batch, features in enumerate(test_data):
                tags = features.pop('tags')
                tags = self.vocab_tags.lookup(tags)
                self.test_step(features, tags)

                if test_batch == steps:
                    break

        # Dimensionality driven learning specific functions.
        def check_turning_point(w):
            if len(self.test_lid) > w:
                if self.test_lid[-1] > np.mean(self.test_lid[-w:-1]) + np.std(self.test_lid[-w:-1]) * 2:
                    print("TURNING POINT: LID-based policy factor enabled.")
                    self.turning_point = True
                    self.load_weights(self.params['checkpoint_dir'] +
                                      f"/{self.params['name']}_ckpt.tf")
                else:
                    self.save_weights(self.params['checkpoint_dir'] +
                                      f"/{self.params['name']}_ckpt.tf")
            else:
                self.save_weights(self.params['checkpoint_dir'] +
                                  f"/{self.params['name']}_ckpt.tf")
        def print_metrics(epoch, batch, training=None):
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

        def modify_policy_factor(i, T):
            i_T = tf.math.divide(i, T)
            LID_min_LID = tf.math.divide(
                self.test_lid[-1],
                min(self.test_lid[:-1])
            )
            exponent = tf.math.multiply(i_T, LID_min_LID)
            alpha = tf.math.exp(-exponent)

            if alpha < 0:
                alpha = 0
            elif alpha > 1 or tf.math.is_nan(alpha):
                alpha = 1
            else:
                pass

            return tf.cast(alpha, tf.float32)

        def policy_factor_check(w, i, T):
            if not self.turning_point:
                check_turning_point(w)
            elif self.turning_point:
                self.policy_factor = modify_policy_factor(i, T)
            else:
                pass

        def single_epoch(test_every_n):
            test_num_lines = buffer_count(ftags(self.params['datadir'], 'test'))
            five_perc = int(
                np.floor((test_num_lines / self.params['batch_size']) / 20))
            num_lines = buffer_count(ftags(self.params['datadir'], 'train'))
            T = int(np.floor(num_lines/self.params['batch_size']))

            for train_batch, features in enumerate(train_data):
                tags = features.pop('tags')
                tags = self.vocab_tags.lookup(tags)
                self.train_step(features, tags)

                if train_batch % test_every_n == 0:
                    _test(five_perc)
                    self.end_of_epoch_metrics(end=True)
                    print_metrics(0, train_batch, training=False)
                    if self.params['name'] == 'ddl':
                        policy_factor_check(w=10, i=train_batch, T=T)
            _test()
            self.end_of_epoch_metrics(end=True)
            print("\nFINAL METRICS")
            print_metrics(0, train_batch, training=False)

        def multi_epoch(update_metrics_every_n):
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
                    print_metrics(epoch, batch, training=True)

                if batch % T == 0 and batch != 0:
                    _test()
                    self.end_of_epoch_metrics(end=True)
                    print_metrics(epoch, batch, training=False)
                    if self.params['name'] == 'ddl':
                        policy_factor_check(w=5, i=i,
                                            T=T*self.params['epochs'])
                    epoch += 1
                    batch = 1
                else:
                    batch += 1

        self.turning_point = False
        if self.params['epochs'] == 1:
            single_epoch(test_every_n=50)
        elif self.params['epochs'] > 1:
            multi_epoch(update_metrics_every_n=100)


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
        kernel_size : int
            kernel size for the temporal convolution
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

def loss_fn(logits, tags, nwords, crf_params, policy_factor):
    log_likelihood, _ = cust_crf_log_likelihood(
        inputs=logits,
        tag_indices=tags,
        sequence_lengths=nwords,
        policy_factor=policy_factor,
        transition_params=crf_params
    )
    loss = tf.compat.v1.reduce_mean(-log_likelihood)

    return loss



#






