#!/usr/bin/env python3

import numpy as np
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from pathlib import Path


def fit_word2vec(datadir, vector_size=300):
    corpus = LineSentence(f'{datadir}/PLOS_sentences.txt')
    print("fitting word2vec model")
    model = Word2Vec(sentences=corpus, size=vector_size, window=5, min_count=1,
                     workers=4)
    print("saving word2vec model")
    model.save(f'{datadir}/word2vec.model')
    print("done")


def adapt_word2vec(datadir, vector_size=300):
    # Load vocab
    with Path(f'{datadir}/vocab_words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    size_vocab = len(word_to_idx)
    # Array of zeros
    embeddings = np.zeros((size_vocab, vector_size))

    model = Word2Vec.load(f'{datadir}/word2vec.model')

    for word, idx in word_to_idx.items():
        vector = model.wv.get_vector(word)
        embeddings[idx] = vector

    np.savez_compressed(f"{datadir}/word2vec.npz", embeddings=embeddings)
