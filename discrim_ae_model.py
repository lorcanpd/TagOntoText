
import tensorflow as tf
import pandas as pd
from .base_model import CustomModelBase
import tensorflow_addons as tfa
from .discrim_auto_encoder import dAutoEncoder, get_jenks_break
from .custom_crf import aere_crf_log_likelihood


class DAEncModel(CustomModelBase):

    def __init__(self, params):
        super().__init__(params)
        self.autoencoder = dAutoEncoder(params)

    def call(self, data, training=None, **kwargs):
        mask = tf.sequence_mask(data['nwords'])  # Mask added.
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

        if training:
            # batch_lids = self._lid(logits=output, k=5)
            # not_inf = tf.math.is_finite(batch_lids)
            # batch_lids = tf.boolean_mask(batch_lids, not_inf)
            for_lids = output
        else:
            # batch_lids = None
            for_lids = None

        if training:
            output = tf.nn.dropout(x=output,
                                   rate=self.params['dropout'])

        logits = self.dense_layer(output)

        pred_ids, _ = tfa.text.crf.crf_decode(potentials=logits,
                                              transition_params=self.crf_params,
                                              sequence_length=nwords)

        if training:
            # Use auto encoder error to discriminate.
            false_negs = self.encoder_predict(embeddings_copy, pred_ids, mask)
        else:
            false_negs = None

        return logits, pred_ids, for_lids, false_negs, embeddings_copy

    def predict_step(self, features):
        _, pred_ids, _, _, _ = self.call(features, training=False)
        pred_ids = self.reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        return pred_ids

    def encoder_predict(self, embeddings, pred_ids, mask):  # Mask added.
        # Mask empty paddings vectors from jenks break calculation.
        flat_embeddings = tf.boolean_mask(embeddings, mask)  # Mask added.
        recon_errors = self.autoencoder.reconstruction_error(
            flat_embeddings
        )
        divider = get_jenks_break(recon_errors)

        flat_embeddings = tf.reshape(
            embeddings,
            [-1, self.autoencoder.input_size]
        )
        recon_errors = self.autoencoder.reconstruction_error(
            flat_embeddings
        )
        cluster_1 = tf.math.less(recon_errors, divider)
        not_zero = tf.math.greater(recon_errors, 0)
        ae_pos = tf.logical_and(cluster_1, not_zero)
        unflat_ae_pos = tf.reshape(ae_pos,
                                   (self.params['batch_size'], -1))
        pred_neg = tf.equal(pred_ids, tf.cast(self.num_tags, tf.int32))
        ae_pos_pred_neg = tf.math.logical_and(unflat_ae_pos, pred_neg)

        if self.autoencoder.i < 1000:
            self.autoencoder.graph_data.update(
                {
                    self.autoencoder.i: {
                        'jenks': divider.numpy(),
                        'recon_errors': recon_errors.numpy()
                    }
                }
            )
            self.autoencoder.i += 1

        return ae_pos_pred_neg

    def encoder_train(self, embeddings, pred_ids):
        # Train autoencoder on embeddings of tokens predicted to be
        # positives.
        pred_pos = tf.not_equal(pred_ids, tf.cast(self.num_tags, tf.int32))
        pred_pos_vecs = tf.boolean_mask(embeddings, pred_pos)
        self.autoencoder.train_step(pred_pos_vecs)

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            logits, pred_ids, for_lids, false_negs, embs = self.call(
                features,
                training=True
            )
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
        # self.lid_utils['lid'].append(batch_lids)

    def test_step(self, features, tags):
        logits, pred_ids, _, _, _ = self.call(features, training=False)
        loss_value = self._loss_fn(logits,
                                   tags,
                                   features['nwords'],
                                   self.crf_params,
                                   None)

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

        # pd.DataFrame.from_dict(self.autoencoder.graph_data,
        #                        orient='index').to_csv(
        #     "{}/{}_jenks_breaks.csv".format(
        #         self.params['datadir'], self.params['name']
        #     )
        # )

        pd.DataFrame.from_dict(self.autoencoder.graph_data,
                               orient='index').to_pickle(
            "{}/{}_jenks_breaks.pickle".format(
                self.params['datadir'], self.params['name']
            )
        )

