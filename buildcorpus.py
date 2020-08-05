from tensorflow.keras.preprocessing.text import text_to_word_sequence as tokenise


def makeCorpus(filepath, outdir, labels):

    def worker(line, labs):

        def getMatches(line, labs):
            gen = (x for x in labs if ' '+x+' ' in ' '+line+' ')
            return [x for x in gen]

        def getTags(tokens, matches):
            matches.sort(key=len, reverse=True)
            lt = len(tokens)
            tags = [None] * len(tokens)
            count = set()

            for match in matches:
                mt = tokenise(match, lower=False)
                lmt = len(mt)

                for x, t in enumerate(tokens):
                    if x+lmt <= lt and all([tk == tokens[x+i] for
                                            i, tk in enumerate(mt)]):
                        for y in range(lmt):
                            if x+y not in count:
                                tags[x+y] = "B" if y == 0 else "I"
                                count.add(x+y)


            return ['O' if tag is None else tag for tag in tags]

        matches = getMatches(line, labs)
        if matches:
            tokens = tokenise(line, lower=False)
            tags = getTags(tokens, matches)
        else:
            tokens = tags = None

        return tokens, tags

    print("Labelling sentences.")
    wordpath = outdir+"/raw_words.txt"
    tagspath = outdir+"/raw_tags.txt"
    with open(filepath, "r") as fp, open(wordpath, "w") as wp, open(tagspath, "w") as tp:
        vocab = set()
        for line in fp:
            line = line.rstrip('\n')
            tokens, tags = worker(line, labels)
            if tokens and tags:
                vocab.update(tokens)
                assert len(tokens) == len(tags), "Number of tokens and tags don't match"
                for i, t in enumerate(tokens):
                    wp.write(f"{t} ")
                    tp.write(f"{tags[i]} ")
                    # op.write(f"{i+1}\t{t}\t{tags[i]}\n")
                # op.write("\n")
                wp.write("\n")
                tp.write("\n")

    with open(outdir+"/vocab_words.txt", "w") as vw, \
            open(outdir+"/vocab_chars.txt", "w") as vc:
        chars = set()
        for word in vocab:
            vw.write(f"{word} \n")
            chars.update(list(word))

        for char in chars:
            vc.write(f"{char} \n")

    print("Labelling complete.")


# def makeCorpus(filepath, fileoutpath, labels):
#
#     def worker(line, labs):
#
#         def getMatches(line, labs):
#             gen = (x for x in labs if ' '+x+' ' in ' '+line+' ')
#             return [x for x in gen]
#
#         def getTags(tokens, matches):
#             matches.sort(key=len, reverse=True)
#             lt = len(tokens)
#             tags = [None] * len(tokens)
#             count = set()
#
#             for match in matches:
#                 mt = tokenise(match)
#                 lmt = len(mt)
#
#                 for x, t in enumerate(tokens):
#                     if x+lmt <= lt and all([tk == tokens[x + i].lower() for i, tk in enumerate(mt)]):
#                         for y in range(lmt):
#                             if x+y not in count:
#                                 tags[x+y] = "B" if y == 0 else "I"
#                                 count.add(x+y)
#
#
#             return ['O' if tag is None else tag for tag in tags]
#
#         matches = getMatches(line, labs)
#         if matches:
#             tokens = tokenise(line)
#             tags = getTags(tokens, matches)
#         else:
#             tokens = tags = None
#
#         return tokens, tags
#
#     print("Labelling sentences.")
#     with open(filepath, "r") as fp, open(fileoutpath, "w") as op:
#         for line in fp:
#             line = line.rstrip('\n')
#             tokens, tags = worker(line, labels)
#             if tokens and tags:
#                 for i, t in enumerate(tokens):
#                     op.write(f"{i+1}\t{t}\t{tags[i]}\n")
#                 op.write("\n")
#
#     print("Labelling complete.")
