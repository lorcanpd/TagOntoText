
import tensorflow as tf
from .base_model import CustomModelBase
from .custom_crf import ddl_crf_log_likelihood
from .split_train_data import buffer_count
from .stream_dataset import ftags
import numpy as np


class DDLModel(CustomModelBase):

    def __init__(self, params):
        super(CustomModelBase, self).__init__(params)
        self.turning_point = False
        self.policy_factor = tf.cast(1, tf.float32)
        self.all_policy_factors = []

    def train_step(self, features, tags):
        with tf.GradientTape() as tape:
            logits, pred_ids, _ = self.call(features, training=True)
            loss_value = self._loss_fn(logits,
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
        loss_value = self._loss_fn(logits,
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

    def _check_turning_point(self, w):
        if len(self.test_lid) > w:
            if self.test_lid[-1] > np.mean(self.test_lid[-w:-1]) + np.std(
                    self.test_lid[-w:-1]) * 2:
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

    def _policy_factor_check(self, w, i, T):
        self.all_policy_factors.append(self.policy_factor)
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

            if train_batch % test_every_n == 0:
                self._test(five_perc)
                self.end_of_epoch_metrics(end=True)
                self._print_metrics(0, train_batch, training=False)
                self._policy_factor_check(w=10, i=train_batch, T=T)
        self._test(test_data)
        self.end_of_epoch_metrics(end=True)
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

            if batch % update_metrics_every_n == 0:
                self.end_of_epoch_metrics(end=False)
                self._print_metrics(epoch, batch, training=True)

            if batch % T == 0 and batch != 0:
                self._test(test_data)
                self.end_of_epoch_metrics(end=True)
                self._print_metrics(epoch, batch, training=False)
                self._policy_factor_check(w=5, i=i,
                                          T=T*self.params['epochs'])
                epoch += 1
                batch = 1
            else:
                batch += 1
        self.save_weights(self.params['checkpoint_dir'] +
                          f"/{self.params['name']}_ckpt.tf")

    def train_and_eval(self, train_data, test_data):
        if self.params['epochs'] == 1:
            self._single_epoch(train_data, test_data, test_every_n=50)
        elif self.params['epochs'] > 1:
            self._multi_epoch(train_data, test_data, update_metrics_every_n=100)

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
