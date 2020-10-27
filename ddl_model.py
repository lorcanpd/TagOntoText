
import tensorflow as tf
from .base_model import CustomModelBase
from .custom_crf import ddl_crf_log_likelihood
from .split_train_data import buffer_count
from .stream_dataset import ftags
import numpy as np


class DDLModel(CustomModelBase):

    def __init__(self, params):
        super().__init__(params)
        self.turning_point = False
        self.policy_factor = tf.cast(1, tf.float32)
        self.all_policy_factors = []

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            logits, pred_ids, batch_lids = self.call(features, training=True)
            loss_value = self._loss_fn(logits,
                                       tags,
                                       features['nwords'],
                                       self.crf_params,
                                       self.policy_factor)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        weights = tf.cast(tf.sequence_mask(features['nwords']), tf.int32)

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

    def test_step(self, features, tags):
        logits, pred_ids, _ = self.call(features, training=False)
        loss_value = self._loss_fn(logits,
                                   tags,
                                   features['nwords'],
                                   self.crf_params,
                                   self.policy_factor)

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

    def _check_turning_point(self, w):

        if len(self.lid_utils['lid']) > w:
            if self.lid_utils['lid'][-1] > \
                    np.nanmean(self.lid_utils['lid'][-w:-1]) + \
                    np.nanstd(self.lid_utils['lid'][-w:-1]) * 2:
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

    def _modify_policy_factor(self, i, T):
        i_T = tf.math.divide(i, T)
        LID_min_LID = tf.math.divide(
            self.lid_utils['lid'][-1],
            min(self.lid_utils['lid'][:-1])
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

    def _policy_factor_check(self, w, i, T):
        if not self.turning_point:
            self._check_turning_point(w)
        elif self.turning_point:
            self.policy_factor = self._modify_policy_factor(i, T)
        else:
            pass

    def _single_epoch(self, train_data, test_data, test_every_n):
        test_num_lines = buffer_count(ftags(self.params['datadir'], 'test'))
        five_perc = int(
            np.floor((test_num_lines / self.params['batch_size']) / 20))
        num_lines = buffer_count(ftags(self.params['datadir'], 'train'))
        T = int(np.floor(num_lines/self.params['batch_size']))

        for train_batch, features in enumerate(train_data):
            tags = features.pop('tags')
            tags = self.vocab_tags.lookup(tags)
            self.train_step(features, tags)

            if train_batch % 25 == 0:
                self.lid_utils['lid'].append(self.epoch_train_metrics['lid'].result())
                self.epoch_train_metrics['lid'].reset_states()
                self.lid_utils['iter'].append(train_batch)
                self._policy_factor_check(w=25, i=train_batch, T=T)
                self.lid_utils['policy_factor'].append(self.policy_factor)

            if train_batch % test_every_n == 0:
                self._test(test_data, five_perc)
                self.end_of_epoch_metrics(i=train_batch, end=True)
                self._print_metrics(0, train_batch, training=False)


        self._test(test_data)
        self.end_of_epoch_metrics(i=train_batch, end=True)
        print("\nFINAL METRICS")
        self.print_metrics(0, train_batch, training=False)
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

            if batch % 25 == 0:
                self.lid_utils['lid'].append(self.epoch_train_metrics['lid'].result())
                self.epoch_train_metrics['lid'].reset_states()
                self.lid_utils['iter'].append(i)
                self._policy_factor_check(w=55, i=i,
                                          T=T * self.params['epochs'])
                self.lid_utils['policy_factor'].append(self.policy_factor)

            if batch % update_metrics_every_n == 0:
                self.end_of_epoch_metrics(i=i,end=False)
                self._print_metrics(epoch, batch, training=True)

            if batch % T == 0 and batch != 0:
                self._test(test_data)
                self.end_of_epoch_metrics(i=i, end=True)
                self._print_metrics(epoch, batch, training=False)

                epoch += 1
                batch = 1
                if epoch > self.params['epochs']:
                    break
            else:
                batch += 1
        self.save_weights(self.params['checkpoint_dir'] +
                          f"/{self.params['name']}_ckpt.tf")

    def _loss_fn(self, logits, tags, nwords, crf_params, policy_factor):
        log_likelihood, _ = ddl_crf_log_likelihood(
            inputs=logits,
            tag_indices=tags,
            sequence_lengths=nwords,
            policy_factor=policy_factor,
            transition_params=crf_params
        )
        loss = tf.compat.v1.reduce_mean(-log_likelihood)

        return loss
