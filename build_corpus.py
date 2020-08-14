#!/usr/bin/env python3

from tensorflow.keras.preprocessing.text \
    import text_to_word_sequence as tokenise
from unidecode import unidecode


def build_corpus(filepath, outdir, labels):

    def worker(sent, labels_):

        def get_matches(line_, labs_):
            gen = (x for x in labs_ if ' '+x+' ' in ' '+line_+' ')
            return [x for x in gen]

        def get_tags(tokens_, matches_):
            matches_.sort(key=len, reverse=True)
            lt = len(tokens_)
            taglist = [None] * lt
            count = set()

            for match in matches_:
                mt = tokenise(match, lower=False)
                lmt = len(mt)

                for x, _ in enumerate(tokens):

                    if x+lmt <= lt and all(
                            [tk == tokens[x+j] for j, tk in enumerate(mt)]
                    ):

                        for y in range(lmt):

                            if x+y not in count:
                                taglist[x+y] = "B" if y == 0 else "I"
                                count.add(x+y)

            return ['O' if tag is None else tag for tag in taglist]

        matches = get_matches(sent, labels_)

        if matches:
            token_list = tokenise(sent, lower=False)
            tag_list = get_tags(token_list, matches)
        else:
            token_list = tag_list = None

        return token_list, tag_list

    print("Labelling sentences.")
    wordpath = f"{outdir}/raw_words.txt"
    tagspath = f"{outdir}/raw_tags.txt"

    with open(filepath, "r") as fp, open(wordpath, "w") as wp, \
            open(tagspath, "w") as tp:

        for line in fp:
            line = unidecode(line).rstrip('\n')
            tokens, tags = worker(line, labels)

            if tokens and tags:
                assert len(tokens) == len(tags),\
                    "Number of tokens and tags don't match"

                for i, t in enumerate(tokens):
                    wp.write(f"{t} ")
                    tp.write(f"{tags[i]} ")

                wp.write("\n")
                tp.write("\n")

    print("Labelling complete.")
