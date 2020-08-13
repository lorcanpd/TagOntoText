#!/usr/bin/env python3

from random import shuffle, seed
from os.path import dirname
from pathlib import Path
from numpy import floor

def bufferCount(filename):
    """Rapidly count the number of lines in a text document"""
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read  # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines

def getRandInd(index_list, num_indexes):
    output = []
    shuffle(index_list)
    for i in range(num_indexes):
        output.append(index_list.pop(-i))

    return set(output)

def refineFile(raw_file, indecies, name):
    """Extract specific lines from the raw text or tag file and append them to
    a new test or validation file."""
    dirpath = dirname(raw_file)
    name2 = Path(raw_file).name.replace('raw_', '').replace('.txt', '')

    with open(raw_file, 'r') as raw, \
            open(f"{dirpath}/{name}_{name2}.txt", 'w') as out:
        for i, line in enumerate(raw):
            if i in indecies:
                out.write(line)

    return f"{name}_{name2}.txt complete"



def splitTrainTestVal(datadir, test_prop, val_prop, seed_=None):

    if seed_ is not None:
        seed(seed_)

    raw_textfile = f'{datadir}/raw_words.txt'
    raw_tagfile = f'{datadir}/raw_tags.txt'

    rawnum = bufferCount(raw_textfile)

    testnum = int(floor(test_prop * rawnum))
    valnum = int(floor(val_prop * rawnum))

    trainlines = [x for x in range(rawnum)]
    testlines = getRandInd(trainlines, testnum)
    vallines = getRandInd(trainlines, valnum)

    shuffle(trainlines)
    refineFile(raw_textfile, trainlines, 'train')
    refineFile(raw_tagfile, trainlines, 'train')

    if testnum > 0:
        refineFile(raw_textfile, testlines, 'test')
        refineFile(raw_tagfile, testlines, 'test')

    if valnum > 0:
        refineFile(raw_textfile, vallines, 'val')
        refineFile(raw_tagfile, vallines, 'val')





