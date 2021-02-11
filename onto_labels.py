#!/usr/bin/env python3


import owlready2
import re
import numpy as np
from nltk.corpus import stopwords
from regex import sub


class OntoLabels:
    
    def __init__(self):
        self.labels = set()
        self.ontoURIs = set()

    @staticmethod
    def local_onto_path(path):
        owlready2.onto_path.append(path)

    def __get_labels(self, uri):
        onto = owlready2.get_ontology(uri).load()
        for cla in onto.classes():
            gen_epr = (
                i for i in
                [x for x in cla.label] +
                [x for x in cla.hasExactSynonym] +
                [x for x in cla.hasBroadSynonym] +
                [x for x in cla.hasRelatedSynonym]
            )
            for x in gen_epr:
                self.labels.add(x)
        self.ontoURIs.add(uri)

    def __format_labels(self):
        pat_1 = re.compile(r"\([^)]*\)")
        pat_2 = re.compile(r"([/]+)(?=[A-Z])", flags=re.I)

        self.labels = {
            pat_2.sub(" \\1 ", pat_1.sub("", x).strip()) for x in self.labels
        }

    def __get_n_grams(self, num=4):
        self.n_grams = set()
        # breakpoint()
        for n in range(1, num):
            for x in self.labels:
                if len(x.split()) > n:
                    for i in range(len(x.split()) - 1):
                        self.n_grams.add(" ".join(x.split()[i:i + n]))
            # {
            #     self.n_grams.add(" ".join(x.split()[i:i + n]))
            #     for i in range(len(x.split()) - 1)
            #     for x in self.labels if len(x.split()) > n
            # }
        # self.n_grams = n_grams

    def __get_matches(self, line):
        gen = (x for x in self.n_grams if ' '+x+' ' in ' '+line+' ')
        return [x for x in gen]

    def __inv_n_gram_freq(self):
        counts = {}
        lines = iter("\n".join(self.labels).splitlines())

        for line in lines:
            matches = self.__get_matches(line)
            for n_gram in matches:
                if n_gram in counts:
                    counts[n_gram] += 1
                else:
                    counts[n_gram] = 1

        log_inv_freq = {k: np.log(1) - np.log(v) for k, v in counts.items()}

        self.top_end = {k: v for k, v in log_inv_freq.items() if v > np.log(0.9)}


    def add_onto_labels(self, uris):
        if type(uris) is list and all([type(i) is str for i in uris]):
            for uri in uris:
                if uri not in self.ontoURIs:
                    print(f"Extracting ontology labels from {uri}.")
                    self.__get_labels(uri)
                else:
                    print(f"Labels already extracted from {uri}.")
                    
        elif type(uris) is str:
            if uris not in self.ontoURIs:
                print(f"Extracting ontology labels from {uris}.")
                self.__get_labels(uris)
            else:
                print(f"Labels already extracted from {uris}.")
        
        else:
            raise TypeError(
                "URIs must be a single URI string or a list of URI strings."
            )

        print("Formatting labels")
        self.__format_labels()
        print("Getting n-grams")
        self.__get_n_grams()
        self.__inv_n_gram_freq()
        self.labels = self.labels.union(self.top_end)
        del self.top_end, self.n_grams

        # Remove common short strings.
        self.labels.difference_update([
            ".", ",", "-", "!", "?",
            # Numbers.
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            # One letter.
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            # Two letter words.
            "of", "to", "in", "it", "is", "be", "as", "at", "so", "we", "he",
            "by", "or", "on", "do", "if", "me", "my", "up", "an", "go", "no",
            "us", "am", "et", "al", "eg",
            # Three letter words.
            "the", "and", "for", "are", "but", "not", "you", "all", "any",
            "can", "had", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "man", "new", "now", "old", "see",
            "two", "way", "who", "boy", "did", "its", "let", "put", "say",
            "she", "too", "use", "per", "set", "max", "min", "der",
            # Other.
            "that", "this", "quality"
        ])
        self.labels.difference_update(set(stopwords.words('english')))

        print("Labels extracted.")
