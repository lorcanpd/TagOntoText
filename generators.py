#!/usr/bin/env python3

# Adapting the code from https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/main.py

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

from masked_conv import masked_conv1d_and_max


# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

def generator(sent_file, tag_file):
    with Path(sent_file).open("r") as sents, Path(tag_file).open("r") as tags:
        for line_sents, line_tags in zip(sents, tags):
            yield parser(line_sents, line_tags)


def parser(line_sents, line_tags):
    # Words and tags.
    words = [w.encode() for w in line_sents.strip("\n").split()]
    tags = [t.encode() for t in line_tags.strip("\n").split()]
    assert len(words) == len(tags), "The number of words and sentences are not equal."

    # Characters.
    chars = [[c.encode() for c in w] for w in line_sents.strip("\n").split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b"<pad>"] * (max_len - 1) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags



def inputter(wordpath, tagpath, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()),  # words, num_words
              ([None, None], [None]),
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator, wordpath, tagpath),
        output_shapes=shapes,
        output_types=types
    )

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1)
               )
    return dataset

def modeller(features, labels, mode, params):

    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read in vocabularies and inputs.
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets']
    )
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets']
    )
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char embeddings.
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embedding', [num_chars + 1, params['dim_chars']], tf.float32
    )
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)

    # Char 1d convolution.
    # Masking tells tensorflow to ignore the padded elements.
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filter'], params['kernel_size']
    )

    # Word embeddings.
    word_ids = vocab_words.lookup(words)
    # Need to download glove!!!
    glove = np.load(params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char embeddings.
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM.
    # Time-major format (sequence_num, batch_size, features) as opposed to
    # (batch_size, sequence_num, features)
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    # axis = -1 means that each words forward and backward lstm outputs are
    # concatenated.
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1,0,2])  # Swap sequence_num and batch_size.
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF.
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [numtags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions.
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags']
        )
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # Loss.
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics.
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights)
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics
            )

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op
            )

def run(params=None):
    if params = None:
        params = {
            'dim_chars': 100,
            'dim': 300,
            'dropout': 0.5,
            'num_oov_buckets': 1,
            'epochs': 25,
            'batch_size': 20,
            'buffer': 15000,
            'filters': 50,
            'kernel_size': 3,
            'lstm_size': 100,
            'words': str(Path("Sandbox/vocab_words")),  # CREATE VOCAB ETC.
            'chars': str(Path("Sandbox/vocab_chars")),
            'tags': str(Path("Sandbox/vocab_tags")),
            'glove': str(Path("../../../../Data/glove.840B.300d.txt"))
        }

    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, f"{name}_words.txt"))

    def ftags(name):
        return str(Path(DATADIR, f"{name}_tags.txt"))

    # Estimator, train, and evaluate.
    train_inpf = functools.partial(inputter,
                                   fwords('train'),
                                   ftags('train'),
                                   params,
                                   shuffle_and_repeat=True)
    eval_inpf = functools.partial(inputter,
                                  fwords('testa'),
                                  ftags('testa'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(modeller, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=200
    )
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def write_predictions(name):
        Path("results/score").mkdir(parents=True, exist_ok=True)
        with Path(f'results/score/{name}_preds.txt').open('wb') as f:
            test_inpf = functools.partial(inputter,
                                          fwords(name),
                                          ftags(name))
            golds_gen = generator(fwords(name),
                                  ftags(name))
            preds_gen = estimator.predict(test_inpf)

            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for words, tag, tag_pred in zin(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'testa', 'testb']:
        write_predictions(name)


















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
