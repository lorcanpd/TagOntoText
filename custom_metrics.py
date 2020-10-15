
import tensorflow as tf


class MultiClassTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='multiclass_recall', **kwargs):
        super(MultiClassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives',
                                              initializer='zeros')

    def update_state(self, y_true, y_pred, num_classes, positive_inds,
                     sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        for i in range(num_classes):
            if i in positive_inds:
                # Calculate true positives.
                ix = tf.cast(i, tf.int32)
                true = tf.equal(y_true, ix)
                pred = tf.equal(y_pred, ix)
                tp_bool = tf.math.logical_and(true, pred)
                if sample_weight is not None:
                    tp_bool = tf.multiply(tf.cast(tp_bool, tf.int32),
                                          sample_weight)
                tp_bool = tf.cast(tp_bool, tf.float32)
                self.true_positives.assign_add(tf.reduce_sum(tp_bool))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # Used to reset metric before the start of an epoch.
        self.true_positives.assign(0.)


class MultiClassFalsePositives(tf.keras.metrics.Metric):

    def __init__(self, name='multiclass_recall', **kwargs):
        super(MultiClassFalsePositives, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name='true_positives',
                                               initializer='zeros')

    def update_state(self, y_true, y_pred, num_classes, positive_inds,
                     sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        for i in range(num_classes):
            if i not in positive_inds:
                # Calculate false positives.
                ix = tf.cast(i, tf.int32)
                true = tf.equal(y_true, ix)
                pred = tf.not_equal(y_pred, ix)
                fp_bool = tf.math.logical_and(true, pred)
                if sample_weight is not None:
                    fp_bool = tf.multiply(tf.cast(fp_bool, tf.int32),
                                          sample_weight)
                fp_bool = tf.cast(fp_bool, tf.float32)
                self.false_positives.assign_add(tf.reduce_sum(fp_bool))

    def result(self):
        return self.false_positives

    def reset_states(self):
        # Used to reset metric before the start of an epoch.
        self.false_positives.assign(0.)


class MultiClassFalseNegatives(tf.keras.metrics.Metric):

    def __init__(self, name='multiclass_recall', **kwargs):
        super(MultiClassFalseNegatives, self).__init__(name=name, **kwargs)
        self.false_negatives = self.add_weight(name='recall',
                                               initializer='zeros')

    def update_state(self, y_true, y_pred, num_classes, positive_inds,
                     sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        for i in range(num_classes):
            if i not in positive_inds:
                # Calculate false negatives.
                ix = tf.cast(i, tf.int32)
                false = tf.not_equal(y_true, ix)
                pred = tf.equal(y_pred, ix)
                fn_bool = tf.math.logical_and(false, pred)
                if sample_weight is not None:
                    fn_bool = tf.multiply(tf.cast(fn_bool, tf.int32),
                                          sample_weight)
                fn_bool = tf.cast(fn_bool, tf.float32)
                self.false_negatives.assign_add(tf.reduce_sum(fn_bool))

    def result(self):
        return self.false_negatives

    def reset_states(self):
        # Used to reset metric before the start of an epoch.
        self.false_negatives.assign(0.)


def precision_fn(true_positives, false_positives):
    precision = tf.math.divide(true_positives, true_positives + false_positives)

    if tf.math.is_nan(precision):
        precision = tf.cast(0, tf.float32)

    return precision


def recall_fn(true_positives, false_negatives):
    recall = tf.math.divide(true_positives, true_positives + false_negatives)

    if tf.math.is_nan(recall):
        recall = tf.cast(0, tf.float32)

    return recall


def f1_fn(true_positives, false_positives, false_negatives):
    fp_fn = tf.math.divide(false_positives + false_negatives, 2)

    if tf.math.is_nan(fp_fn):
        fp_fn = tf.cast(0, tf.float32)

    f1 = tf.math.divide(true_positives, true_positives + fp_fn)

    if tf.math.is_nan(f1):
        f1 = tf.cast(0, tf.float32)

    return f1
