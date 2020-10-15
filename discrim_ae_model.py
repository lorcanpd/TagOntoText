
import tensorflow as tf
from .base_model import CustomModelBase
import tensorflow_addons as tfa
from .discrim_auto_encoder import dAutoEncoder, get_jenks_break
from .custom_crf import aere_crf_log_likelihood


class DAEncModel(CustomModelBase):

    def __init__(self, params):
        super().__init__(params)
        self.autoencoder = dAutoEncoder(params)

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

        embeddings_copy = embeddings

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

        if training:
            # Use auto encoder error to discriminate.
            # breakpoint()
            false_negs = self.encoder_predict(embeddings_copy, pred_ids)
        else:
            false_negs = None

        return logits, pred_ids, batch_lids, false_negs, embeddings_copy

    def encoder_predict(self, embeddings, pred_ids):
        flat_embeddings = tf.reshape(
            embeddings,
            [-1, self.autoencoder.input_size]
        )
        recon_errors = self.autoencoder.reconstruction_error(
            flat_embeddings
        )
        divider = get_jenks_break(recon_errors)
        ae_pos = tf.math.less(recon_errors, divider)
        unflat_ae_pos = tf.reshape(ae_pos,
                                   (self.params['batch_size'], -1))
        pred_neg = tf.equal(pred_ids, tf.cast(self.num_tags, tf.int32))
        ae_pos_pred_neg = tf.math.logical_and(unflat_ae_pos, pred_neg)

        return ae_pos_pred_neg

    def encoder_train(self, embeddings, pred_ids):
        # Train autoencoder on embeddings of tokens predicted to be
        # positives.
        pred_pos = tf.not_equal(pred_ids, tf.cast(self.num_tags, tf.int32))
        pred_pos_vecs = tf.boolean_mask(embeddings, pred_pos)
        self.autoencoder.train_step(pred_pos_vecs)

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            logits, pred_ids, _, false_negs, embs = self.call(features,
                                                              training=True)
            loss_value = self._loss_fn(logits,
                                       tags,
                                       features['nwords'],
                                       self.crf_params,
                                       false_negs)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_weights)

        self.optimizer.apply_gradients(
            (g, v) for (g, v) in zip(grads, self.trainable_weights)
            if g is not None
        )

        self.encoder_train(embs, pred_ids)

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
        logits, pred_ids, batch_lids, _, _ = self.call(features, training=False)
        loss_value = self._loss_fn(logits,
                                   tags,
                                   features['nwords'],
                                   self.crf_params,
                                   None)

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

    def _loss_fn(self, logits, tags, nwords, crf_params, false_negatives):
        log_likelihood, _ = aere_crf_log_likelihood(
            inputs=logits,
            tag_indices=tags,
            sequence_lengths=nwords,
            false_negatives=false_negatives,
            transition_params=crf_params
        )
        loss = tf.compat.v1.reduce_mean(-log_likelihood)

        return loss
