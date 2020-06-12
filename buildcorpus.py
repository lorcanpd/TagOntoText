from tensorflow.keras.preprocessing.text import text_to_word_sequence as tokenise

def makeCorpus(filepath, fileoutpath, labels):

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
                mt = tokenise(match)
                lmt = len(mt)

                for x, t in enumerate(tokens):
                    if x+lmt <= lt and all([tk == tokens[x + i].lower() for i, tk in enumerate(mt)]):
                        for y in range(lmt):
                            if x+y not in count:
                                count.add(x+y)
                                tags[x+y] = "B" if y == 0 else "I"

            return ['O' if tag is None else tag for tag in tags]

        matches = getMatches(line, labs)
        if matches:
            tokens = tokenise(line)
            tags = getTags(tokens, matches)
        else:
            tokens = tags = None

        return tokens, tags

    print("Labelling sentences.")
    with open(filepath, "r") as fp, open(fileoutpath, "w") as op:
        for line in fp:
            tokens, tags = worker(line, labels)
            if tokens and tags:
                for i, t in enumerate(tokens):
                    op.write(f"{i}\t{t}\t{tags[i]}\n")
    print("Labelling complete.")
