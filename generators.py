#!/usr/bin/env python3

# Adapting the code from https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/main.py

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

# print(tf.executing_eagerly())


# Logging
Path('Results').mkdir(exist_ok=True)
# tf.logging.set_verbosity(logging.INFO)
tf.compat.v1.logging.set_verbosity(logging.INFO)


handlers = [
    logging.FileHandler('Results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

def generator(sent_file, tag_file):
    # with Path(sent_file).open("r") as sents, Path(tag_file).open("r") as tags:
    with open(sent_file, "r") as sents, open(tag_file, "r") as tags:
        for line_sents, line_tags in zip(sents, tags):
            # output = parser(line_sents, line_tags)
            # if output is None:
            #     breakpoint()
            #     pass
            # else:
            #     yield output
            yield parser(line_sents, line_tags)


def parser(line_sents, line_tags):
    # Words and tags.
    words = [w.encode() for w in line_sents.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    # assert len(words) == len(tags), "The number of words and sentences are not equal."
    if len(words) != len(tags):
        breakpoint()

    # Characters.
    chars = [[c.encode() for c in w] for w in line_sents.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b"<pad>"] * (max_len - l) for c, l in zip(chars, lengths)]


    # breakpoint()

    return (((words, len(words)), (chars, lengths)), tags)



def inputter(wordpath, tagpath, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    # shapes = ((([None], ()),  # words, num_words
    #            ([None, None], [None])),
    #           [None])  # tags

    shapes = (((tf.TensorShape(dims=[None]), tf.TensorShape(dims=())),  # words, num_words
               (tf.TensorShape(dims=[None, None]), tf.TensorShape(dims=[None]))),
              tf.TensorShape(dims=[None]))  # tags


    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)


    # defaults = ((('<pad>', 0),
    #              ('<pad>', 0)),
    #             'O')

    # defaults = ((('<pad>', tf.constant(0)),
    #              ('<pad>', tf.constant(0))),
    #             'O')




    # dataset = tf.data.Dataset.from_generator(
    #     functools.partial(generator, wordpath, tagpath),
    #     output_shapes=shapes,
    #     output_types=types
    # )
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_shapes=shapes,
        output_types=types,
        args=(wordpath, tagpath)
    )




    def unpack(w_lw_c_lc, t):
        # breakpoint()
        # w_lw_c_lc, t = w_lw_c_lc
        w_lw, c_lc = w_lw_c_lc
        w, lw = w_lw
        c, lc = c_lc

        return {'words': w, 'nwords': lw, 'chars': c, 'nchars': lc, 'tags': t}

    dataset = dataset.map(map_func=unpack)


    # breakpoint()

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
    # breakpoint()
    dataset = (dataset
    # ds = (dataset
               # .padded_batch(params.get('batch_size', 20), shapes, defaults)
               # .batch(batch_size=params.get('batch_size', 20))
               .padded_batch(batch_size=params.get('batch_size', 20),
               # .padded_batch(batch_size=params.get('batch_size', 20),
               #               padded_shapes=shapes,
                             padded_shapes=padded_shapes,
                             # padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
                             padding_values=defaults)
               .prefetch(1)
               )

    # breakpoint()
    return dataset

# @tf.function
# May have to delete labels....?
def modeller(features,
             # labels,
             mode, params):

    if isinstance(features, dict):
        # breakpoint()
        labels = features['tags']
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read in vocabularies and inputs.
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    # nwords = nwords[0]
    # breakpoint()
    training = (mode == tf.estimator.ModeKeys.TRAIN)

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


    # vocab_words = tf.contrib.lookup.index_table_from_file(
    #     params['words'], num_oov_buckets=params['num_oov_buckets']
    # )

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

    # vocab_chars = tf.contrib.lookup.index_table_from_file(
    #     params['chars'], num_oov_buckets=params['num_oov_buckets']
    # )
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char embeddings.
    char_ids = vocab_chars.lookup(chars)
    # variable = tf.get_variable(
    #     name='chars_embedding', shape=[num_chars + 1, params['dim_chars']],
    #     dtype=tf.float32
    # )
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
    # char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
    #                                     training=training)
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
    # dim_x, dim_y = glove.shape
    # breakpoint()
    # glove = tf.keras.initializers.Constant(glove)
    variable = tf.Variable(
        variable,
        shape=variable.shape,
        dtype=tf.float32,
        trainable=False
    )
    # breakpoint()
    # word_embedding = tf.keras.layers.Embedding(
    #     input_dim=dim_x,
    #     output_dim=dim_y,
    #     embeddings_initializer=glove
    # )
    # word_embeddings = word_embedding(words)
    # breakpoint()
    word_ids = vocab_words.lookup(words)
    word_embeddings = tf.nn.embedding_lookup(params=variable, ids=word_ids)
    # breakpoint()

    # Concatenate Word and Char embeddings.
    # breakpoint()
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    # breakpoint()
    # embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
    if training:
        embeddings = tf.nn.dropout(x=embeddings, rate=dropout)

    # breakpoint()

    # LSTM.
    # Time-major format (sequence_num, batch_size, features) as opposed to
    # (batch_size, sequence_num, features)

    # t = tf.transpose(embeddings, perm=[1, 0, 2])
    # t = embeddings

    # breakpoint()

    # lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    # lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    # lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    # output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    # output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    #
    # tf2 version:
    # lstm_fw = tf.keras.layers.LSTM(
    #     units=tf.keras.layers.LSTMCell(params['lstm_size']),
    #     time_major=True
    # )
    # lstm_bw = tf.keras.layers.LSTM(
    #     units=tf.keras.layers.LSTMCell(params['lstm_size']),
    #     time_major=True,
    #     go_backwards=True
    # )
    # breakpoint()
    # bi_lstm = tf.keras.layers.Bidirectional(
    #     layer=tf.keras.layers.LSTM(
    #         units=params['lstm_size']
    #         # time_major=True
    #     ),
    #     merge_mode='concat'
    #     # input_shape=(nwords, nwords, embeddings.shape[2])
    # )
    # output = bi_lstm(t)

    # breakpoint()

    output = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=params['lstm_size'],
            return_sequences=True
            # time_major=True
        ),
        merge_mode='concat'#,
        # input_shape=(None, None, embeddings.shape[2])
    )(embeddings) #(t)

    # bi_lstm = tf.keras.layers.Bidirectional(
    #     layer=tf.keras.layers.LSTM(
    #         units=params['lstm_size'],
    #         return_sequences=True
    #         # time_major=True
    #     ),
    #     input_shape=(params.get('batch_size', 20), nwords, embeddings.shape[2])
    # )
    #
    # output = bi_lstm(t)


    # lstm_fw = tf.keras.layers.LSTM(
    #     units=params['lstm_size'],
    #     time_major=True
    # )
    # lstm_bw = tf.keras.layers.LSTM(
    #     units=params['lstm_size'],
    #     time_major=True,
    #     go_backwards=True
    # )

    # lstm_fw = tf.raw_ops.BlockLSTM(params['lstm_size'])
    #
    # lstm_bw = tf.raw_ops.BlockLSTM(params['lstm_size'])

    # output_fw = lstm_fw(t)
    # output_bw = lstm_bw(t)

    # LSTM does not need time-major formatting.
    # @tf.function
    # def fw_lstm(inputs):
    #     output, _, _ = lstm_fw(inputs=inputs)
    #     return output
    #
    # # @tf.function
    # def bw_lstm(inputs):
    #     output, _, _ = lstm_bw(inputs=inputs)
    #     return output

    # output_fw = fw_lstm(inputs=t)
    # output_bw = bw_lstm(inputs=t)

    # breakpoint()


    # axis = -1 means that each words forward and backward lstm outputs are
    # concatenated.
    # output = tf.concat([output_fw, output_bw], axis=-1)
    # output = tf.transpose(output, perm=[1,0,2])  # Swap sequence_num and batch_size.
    # output = tf.layers.dropout(output, rate=dropout, training=training)
    # output = tf.keras.layers.dropout(output, rate=dropout, training=training)
    if training:
        output = tf.nn.dropout(x=output, rate=dropout)



    # CRF.
    # logits = tf.layers.dense(output, num_tags)
    logits = tf.keras.layers.Dense(num_tags)(output)
    # crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)

    crf_params = tf.Variable(
        # initial_value=tf.zeros(
        #     # shape=[num_tags, num_tags],
        #     shape=[num_tags, num_tags],
        #     dtype=tf.float32
        # ),
        initial_value=tf.random.uniform(
            shape=[num_tags, num_tags],
            minval=0,
            maxval=1,
            dtype=tf.float32
        ),
        # initial_value=None,
        name="crf",
        # shape=[num_tags, num_tags],
        shape=(num_tags, num_tags),
        dtype=tf.float32
    )

    # pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
    # breakpoint()
    pred_ids, _ = tfa.text.crf.crf_decode(potentials=logits,
                                          transition_params=crf_params,
                                          sequence_length=nwords)
    # breakpoint()
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions.

        # reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
        #     params['tags']
        # )
        # breakpoint()

        # reverse_vocab_tags = tf.lookup.StaticVocabularyTable(
        #     tf.lookup.TextFileInitializer(
        #         filename=params['tags'],
        #         # Might have to swap keys and value stuff around...
        #         key_dtype=tf.int64,
        #         key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        #         value_dtype=tf.string,
        #         value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        #         delimiter="\n"
        #     ),
        #     num_oov_buckets=params['num_oov_buckets'],
        #     lookup_key_dtype=tf.int64
        # )

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

        # breakpoint()

        # pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        pred_strings = reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # Loss.
        # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
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
        # tags = vocab_tags.lookup(labels)

        # breakpoint()

        tags = vocab_tags.lookup(labels)
        # log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
        #     logits, tags, nwords, crf_params
        # )
        log_likelihood, _ =  tfa.text.crf_log_likelihood(
            inputs=logits,
            tag_indices=tags,
            sequence_lengths=nwords,
            transition_params=crf_params
        )
        loss = tf.compat.v1.reduce_mean(-log_likelihood)

        # Metrics.
        weights = tf.sequence_mask(nwords)
        metrics = {
            # 'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            # 'acc': tf.keras.metrics.Accuracy().update_state(
            #     y_true=tags,
            #     y_pred=pred_ids,
            #     sample_weight=weights
            # ),
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
            # breakpoint()
            # tf.summary.scalar(metric_name, op[1])
            tf.compat.v1.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics
            )


        elif mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = tf.train.AdamOptimizer().minimize(
            #     loss, global_step=tf.train.get_or_create_global_step()
            # )
            # breakpoint()
            train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op
            )
        # breakpoint()

def run(DATADIR, params=None):
    DATADIR = str(Path(DATADIR))
    if params is None:
        params = {
            'dim_chars': 100,
            'dim': 300,
            'dropout': 0.5,
            'num_oov_buckets': 1,
            'epochs': 1,
            'batch_size': 20,
            # 'batch_size': 2,
            'buffer': 15000,
            'filter': 50,
            'kernel_size': 3,
            # 'kernel_size': 1,
            'lstm_size': 100,
            'words': str(Path("Sandbox/vocab_words.txt")),
            'chars': str(Path("Sandbox/vocab_chars.txt")),
            'tags': str(Path("Sandbox/vocab_tags.txt")),
            'glove': str(Path("Sandbox/glove.npz"))
        }

    with Path('Results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(datadir, name):
        return str(Path(datadir, f"{name}_words.txt"))

    def ftags(datadir, name):
        return str(Path(datadir, f"{name}_tags.txt"))

    # Estimator, train, and evaluate.
    train_inpf = functools.partial(inputter,
                                   fwords(DATADIR, 'train'),
                                   ftags(DATADIR, 'train'),
                                   params,
                                   shuffle_and_repeat=True)
    # train_inpf = inputter(fwords('train'), ftags('train'),
                          # params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(inputter,
                                  fwords(DATADIR, 'test'),
                                  ftags(DATADIR, 'test'))

    # eval_inpf = inputter(fwords('test'), ftags('test'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn=modeller,
                                       model_dir='Results/model',
                                       config=cfg,
                                       params=params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    # hook = tf.contrib.estimator.stop_if_no_increase_hook(
    #     estimator, 'f1', 500, min_steps=8000, run_every_secs=200
    # )
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
    # eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf)
    # breakpoint()
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def write_predictions(datadir, name):
        Path("Results/score").mkdir(parents=True, exist_ok=True)
        with Path(f'Results/score/{name}_preds.txt').open('wb') as f:
            test_inpf = functools.partial(inputter,
                                          fwords(datadir, name),
                                          ftags(datadir, name))
            golds_gen = generator(fwords(datadir, name),
                                  ftags(datadir, name))
            preds_gen = estimator.predict(test_inpf)

            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for words, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([words, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'test', 'val']:
        write_predictions(DATADIR, name)


















# Test generator.
# for line, tags in generator("Sandbox/test_words.txt", "Sandbox/test_tags.txt"):
#     print(tags)
#     print(line)



# Testing.
# dataset = tf.data.Dataset.from_generator(
#     functools.partial(generator, "Sandbox/test_words.txt", "Sandbox/test_tags.txt"),
#     output_shapes=shapes, output_types=types
# )
#
#
# for tf_words, tf_size in dataset:
#     print(tf_words, tf_size)

#
# def ELMoEmbedder(input_text):
#     return elmo(tf.reshape(tf.cast(input_text, tf.string),
#                            [-1]), signature="tokens", as_dict=False)
#
# ELMoEmbedder("hello")
#
#
#
#
# import tensorflow_hub as hub
#
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
#
# # elmo.get_signature_names()
# # elmo.get_input_info_dict(signature='tokens')
# embeddings = elmo(dict(text=dataset))
#
#
#
# test = {"a": 2,
#         "b": 3}
#
# test.get("a", 20)
#
# params = {"epochs": 1,
#           "buffer": 10,
#           "batch_size": 2}
#
# # Need to disable eager execution to run Session() as below.
# tf.compat.v1.disable_eager_execution()
#
# dataset = inputter("Sandbox/test_words.txt", "Sandbox/test_tags.txt", params)
# iterator = dataset.make_one_shot_iterator()
# node = iterator.get_next()
#
# with tf.compat.v1.Session() as sesh:
#     print(sesh.run(node))
#
#






























#
# from pathlib import Path
# import functools
# import itertools
# import random
# import numpy as np
# import tensorflow as tf
# import os
#
# def training_parser(sent_words, sent_tags):
#     words = [w.encode() for w in sent_words.strip().split()]
#     tags = [t.encode() for t in sent_tags.strip().split()]
#     assert len(words) == len(tags), "The number of words and sentences are not equal"
#
#     return (words, len(words)), tags
#
# def training_generator(wordpath, tagpath):
#     with Path(wordpath).open('r') as wordfile, Path(tagpath).open('r') as tagfile:
#         for sent_words, sent_tags in zip(wordfile, tagfile):
#             yield training_parser(sent_words, sent_tags)
#
# def inputs(words, tags, params=None, shuffle_and_repeat=False):
#     params = params if params is not None else {}
#     shapes = (([None], ()),  # words, num_words
#               [None])        # tags
#     types = ((tf.string, tf.int32),
#               tf.string)
#     defaults = (('<pad>', 0),
#                 'O')
#     dataset = tf.data.Dataset.from_generator(
#         functools.partial(training_generator, words, tags),
#         output_shapes=shapes, output_types=types)
#
#     if shuffle_and_repeat:
#         dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
#
#     dataset = (dataset
#                .padded_batch(params.get('batch_size', 20), shapes,defaults)
#                .prefetch(1))
#     return dataset
#
#
#
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
#
# tokens = [['the', 'first', 'time', 'ever', 'I', 'saw', 'your', 'face']]
#
# tokens_length = [8]
#
# embeddings = elmo(inputs={"tokens": tokens,
#                           "sequence_len": tokens_length},
#                   signature="tokens",
#                   as_dict=True)["word_emb"]
#
#
# def get_indecies(filepath):
#     with open(filepath, "r") as data:
#         # Get number of lines.
#         for i, l in enumerate(data):
#             pass
#         return [x for x in range(0, i+1)]
#
# def linegen(filepath):
#     for line in open(filepath, "r"):
#         yield line
#
# def getLines(filepath, batch_indecies):
#     lines = []
#     for i, line in enumerate(linegen(filepath)):
#         if i in batch_indecies:
#             lines.append(line)
#     return lines
#
#
# filepath = "Sandbox/raw_word.txt"
# batch_size = 5
# indecies = get_indecies(filepath)
# random.shuffle(indecies)
# batch_indicies = []
# for i in range(0, batch_size):
#     try:
#         batch_indicies.append(indecies.pop())
#     except IndexError:
#         pass
# getLines(filepath, batch_indicies)
#
# # class DataGenerator(tf.data.Dataset):
# class DataGenerator():
#
#     def __init__(self, sentence_file, tag_file, indices, batchsize, shuffle=True):
#         self.sent_file = sentence_file
#         self.tag_file = tag_file
#         self.indices = indices
#         self.batchsize = batchsize
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         return int(np.floor(len(self.indices) / self.batchsize))
#
#     def __getitem__(self, batch):
#         batch_indices = self.indices[batch*self.batchsize:(batch+1)*self.batchsize]
#         words = getLines(self.sent_file, batch_indices)
#         tags = getLines(self.tag_file, batch_indices)
#
#         yield (words, len(words)), tags
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             random.shuffle(self.indices)
#
#     def linegen(self, filepath):
#         for line in open(filepath, "r"):
#             yield line
#
#     def getLines(self, filepath, batch_indices):
#         lines = []
#         for i, line in enumerate(self.linegen(filepath)):
#             if i in batch_indices:
#                 lines.append([x.encode() for x in line.strip().split()])
#                 # lines.append(line)
#         return lines
#
#
#
#
#
# words = getLines(self.sent_file, batch_indices)
# tags = getLines(self.tag_file, batch_indices)
#
# yield (words, len(words)), tags
#
# def on_epoch_end(self):
#     if self.shuffle:
#         random.shuffle(self.indices)
#
# def linegen(self, filepath):
#     for line in open(filepath, "r"):
#         yield line
#
# def getLines(self, filepath, batch_indices):
#     lines = []
#     for i, line in enumerate(self.linegen(filepath)):
#         if i in batch_indices:
#             lines.append([x.encode() for x in line.strip().split()])
#             # lines.append(line)
#     return lines
#
#
#
# #
# # def inputs(wordpath, tagpath, params=None, shuffle=True):
# #
# #
# #     # tf.data.Dataset.from_generator()
# #
# #     params = params if params is not None else {}
# #     shapes = (([None], ()),  # words, num_words
# #               [None])        # tags
# #     types = ((tf.string, tf.int32),
# #               tf.string)
# #     defaults = (('<pad>', 0),
# #                 'O')
# #     dataset = tf.data.Dataset.from_generator(training_generator, wordpath, tagpath),
# #         output_shapes=shapes, output_types=types)
#
# # directory = "Sandbox"
# # FILE_NAMES = ["raw_word.txt"]#, "raw_tag.txt"]
# #
# # for i, file_name in enumerate(FILE_NAMES):
# #     lines_dataset = tf.data.TextLineDataset(os.path.join(directory, file_name))
#
#
# # def data_gen(filepath, batch_size, indecies):
# #     output = []
# #     with open(filepath, "r") as data:
# #         batch_indecies = [indecies.pop() for i in range(batch_size)]
# #         # for i in batch_indecies:
# #         while batch_indecies:
# #             # breakpoint()
# #             i = batch_indecies.pop()
# #             output.append(next(itertools.islice(data, i, i+1)))
# #     yield from output
#
#
#
# # with open("Sandbox/raw_word.txt", "r") as data:
# #     # Get number of lines.
# #     for i, l in enumerate(data):
# #         pass
# #     num_lines = i + 1
# #
# #     batch_size = 10
# #     indecies = [x for x in range(0, num_lines)]
# #
# #     random.shuffle(indecies)
# #
# #     batch_indecies = [indecies.pop() for x in range(batch_size)]
# #
# #     for i in batch_indecies:
# #         yield itertools.islice(data,index_of_interest)
# #     # line = next(
#     #     itertools.islice(data, index_of_interest, index_of_interest + batch_size), None)
#
#
# input_path = "Sandbox/raw_word.txt"
#
# index_of_interest = 300
#
# line = next(itertools.islice(data,index_of_interest,index_of_interest+1),None)
