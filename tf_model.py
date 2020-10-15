#!/usr/bin/env python3

# Adapting the code from:
# https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/main.py

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from .tf_metrics import precision, recall, f1

from .masked_conv import masked_conv1d_and_max
from .custom_crf import crf_log_likelihood as cust_crf_log_likelihood
from .custom_crf import lid

# Logging
Path('Results').mkdir(exist_ok=True)
tf.compat.v1.logging.set_verbosity(logging.INFO)


handlers = [
    logging.FileHandler('Results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def generator(sent_file, tag_file):
    with open(sent_file, "r") as sents, open(tag_file, "r") as tags:
        for line_sents, line_tags in zip(sents, tags):
            yield parser(line_sents, line_tags)


def parser(line_sents, line_tags):
    # Words and tags.
    words = [w.encode() for w in line_sents.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and sentences are not equal length."

    # Characters.
    chars = [[c.encode() for c in w] for w in line_sents.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b"<pad>"] * (max_len - l) for c, l in zip(chars, lengths)]

    return ((words, len(words)), (chars, lengths)), tags


def inputter(wordpath, tagpath, params=None, shuffle_and_repeat=False):

    params = params if params is not None else {}
    # words, nwords, chars, nchar, tags.
    shapes = (((tf.TensorShape(dims=[None]), tf.TensorShape(dims=())),
               (tf.TensorShape(dims=[None, None]),
                tf.TensorShape(dims=[None]))),
              tf.TensorShape(dims=[None]))

    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_shapes=shapes,
        output_types=types,
        args=(wordpath, tagpath)
    )

    def unpack(w_lw_c_lc, t):
        w_lw, c_lc = w_lw_c_lc
        w, lw = w_lw
        c, lc = c_lc

        return {'words': w, 'nwords': lw, 'chars': c, 'nchars': lc, 'tags': t}

    dataset = dataset.map(map_func=unpack)

    padded_shapes = {
        'words': tf.TensorShape(dims=[None]),
        'nwords': tf.TensorShape(dims=[]),
        'chars': tf.TensorShape(dims=[None, None]),
        'nchars': tf.TensorShape(dims=[None]),
        'tags': tf.TensorShape(dims=[None])
    }

    defaults = {
        'words': '<pad>',
        'nwords': tf.constant(0),
        'chars': '<pad>',
        'nchars': tf.constant(0),
        'tags': 'O'
    }

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(batch_size=params.get('batch_size', 20),
                             padded_shapes=padded_shapes,
                             padding_values=defaults)
               .prefetch(1)
               )
    return dataset


def modeller(features, mode, params):
    if isinstance(features, dict):
        labels = features['tags']
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))
    else:
        raise
    # Read in vocabularies and inputs.
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    testing = (mode == tf.estimator.ModeKeys.EVAL)

    vocab_words = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=params['words'],
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\n"
        ),
        num_oov_buckets=params['num_oov_buckets']
    )

    vocab_chars = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=params['chars'],
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\n"
        ),
        num_oov_buckets=params['num_oov_buckets'],
        lookup_key_dtype=tf.string
    )

    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char embeddings.
    char_ids = vocab_chars.lookup(chars)

    variable = tf.Variable(
        initial_value=tf.zeros(
            shape=[num_chars + 1, params['dim_chars']],
            dtype=tf.float32
        ),
        name='chars_embedding',
        shape=[num_chars + 1, params['dim_chars']],
        dtype=tf.float32
    )
    char_embeddings = tf.nn.embedding_lookup(params=variable, ids=char_ids)

    if training:
        char_embeddings = tf.nn.dropout(x=char_embeddings, rate=dropout)

    # Char 1d convolution.
    # Masking tells tensorflow to ignore the padded elements.
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filter'], params['kernel_size']
    )

    # Word embeddings.
    glove = np.load(params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(
        variable,
        shape=variable.shape,
        dtype=tf.float32,
        trainable=False
    )
    word_ids = vocab_words.lookup(words)
    word_embeddings = tf.nn.embedding_lookup(params=variable, ids=word_ids)

    # Concatenate Word and Char embeddings.
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

    if training:
        embeddings = tf.nn.dropout(x=embeddings, rate=dropout)

    # LSTM.
    output = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=params['lstm_size'],
            return_sequences=True
        ),
        merge_mode='concat'
    )(embeddings)

    # ADD LID CALCULATION HERE.
    if testing:
        batch_lids = lid(logits=output, k=20)



    if training:
        output = tf.nn.dropout(x=output, rate=dropout)
    # CRF.
    logits = tf.keras.layers.Dense(num_tags)(output)

    # Randomly initialised transition parameters.
    crf_params = tf.Variable(
        initial_value=tf.random.uniform(
            shape=[num_tags, num_tags],
            minval=0,
            maxval=1,
            dtype=tf.float32
        ),
        name="crf",
        shape=(num_tags, num_tags),
        dtype=tf.float32
    )

    pred_ids, _ = tfa.text.crf.crf_decode(potentials=logits,
                                          transition_params=crf_params,
                                          sequence_length=nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        reverse_vocab_tags = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename=params['tags'],
                key_dtype=tf.int64,
                key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                value_dtype=tf.string,
                value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                delimiter="\n"
            ),
            default_value='O'
        )
        pred_strings = reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # Loss.
        vocab_tags = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=params['tags'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\n"
            ),
            num_oov_buckets=1,
            lookup_key_dtype=tf.string
        )
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tfa.text.crf_log_likelihood(
            inputs=logits,
            tag_indices=tags,
            sequence_lengths=nwords,
            transition_params=crf_params
        )
        # log_likelihood = cust_crf_log_likelihood(
        #     imputs=logits,
        #     tag_indices=tags,
        #     sequence_lengths=nwords,
        #     transition_params=crf_params,
        #     policy_factor=policy_factor
        # )
        loss = tf.compat.v1.reduce_mean(-log_likelihood)

        # Metrics.
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.compat.v1.metrics.accuracy(
                labels=tags,
                predictions=pred_ids,
                weights=weights
            ),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights)
        }
        for metric_name, op in metrics.items():
            tf.compat.v1.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics
            )

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op
            )


def run_train_eval(datadir, params=None):
    if params is None:
        params = {
            'dim_chars': 100,
            'dim': 300,
            'dropout': 0.5,
            'num_oov_buckets': 1,
            'epochs': 1,
            'batch_size': 20,
            'buffer': 15000,
            'filter': 50,
            'kernel_size': 3,
            'lstm_size': 100,
            'words': str(Path(f"{datadir}/vocab_words.txt")),
            'chars': str(Path(f"{datadir}Sandbox/vocab_chars.txt")),
            'tags': str(Path(f"{datadir}/vocab_tags.txt")),
            'glove': str(Path(f"{datadir}/glove.npz"))
        }

    with Path('Results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(filedir, filename):
        return str(Path(filedir, f"{filename}_words.txt"))

    def ftags(filedir, filename):
        return str(Path(filedir, f"{filename}_tags.txt"))

    # Estimator, train, and evaluate.
    train_inpf = functools.partial(inputter,
                                   fwords(datadir, 'train'),
                                   ftags(datadir, 'train'),
                                   params,
                                   shuffle_and_repeat=True)
    eval_inpf = functools.partial(inputter,
                                  fwords(datadir, 'test'),
                                  ftags(datadir, 'test'))
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn=modeller,
                                       model_dir='Results/model',
                                       config=cfg,
                                       params=params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator=estimator,
        metric_name='f1',
        max_steps_without_increase=500,
        min_steps=8000,
        run_every_secs=200
    )
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf,
                                      start_delay_secs=240,
                                      throttle_secs=200)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def write_predictions(filedir, filename):
        Path("Results/score").mkdir(parents=True, exist_ok=True)
        with Path(f'Results/score/{filename}_preds.txt').open('wb') as file:
            test_inpf = functools.partial(inputter,
                                          fwords(filedir, filename),
                                          ftags(filedir, filename))
            golds_gen = generator(fwords(filedir, filename),
                                  ftags(filedir, filename))
            preds_gen = estimator.predict(test_inpf)

            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for words, tag, tag_pred in zip(words, tags, preds['tags']):
                    file.write(b' '.join([words, tag, tag_pred]) + b'\n')
                file.write(b'\n')

    for name in ['train', 'test', 'val']:
        write_predictions(datadir, name)
