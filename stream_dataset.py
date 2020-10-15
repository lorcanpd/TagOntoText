
import tensorflow as tf
from pathlib import Path
import time


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
               # .prefetch(tf.data.experimental.AUTOTUNE)
               )
    return dataset




def fwords(filedir, filename):
    return str(Path(filedir, f"{filename}_words.txt"))

def ftags(filedir, filename):
    return str(Path(filedir, f"{filename}_tags.txt"))



# Function for benchmarking data streaming.
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)