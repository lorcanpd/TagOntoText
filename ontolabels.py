#!/usr/bin/env python3


import owlready2
from regex import sub


class OntoLabels:
    
    def __init__(self):
        
        self.labels = set()
        self.ontoURIs = set()

    def localOntoPath(self, path):
        owlready2.onto_path.append(path)

    def __getLabels(self, uri):
        onto = owlready2.get_ontology(uri).load()
        for c in onto.classes():
            try:
                # lab = sub(r"\s*(\((?>[^()]+|(?1))*\))$", "", c.label[0])
                self.labels.add(sub(r"\s*(\((?>[^()]+|(?1))*\))$", "",
                                    c.label[0]))
                # self.labels.add(c.label[0])
            except IndexError:
                pass
            # except OwlReadyOntologyParsingError:
            #     continue
        self.ontoURIs.add(uri)
            
    def addOntoLabels(self, uris):
        if type(uris) is list and all([type(i) is str for i in uris]):
            for uri in uris:
                if uri not in self.ontoURIs:
                    print(f"Extracting ontology labels from {uri}.")
                    self.__getLabels(uri)
                else:
                    print(f"Labels already extracted from {uri}.")
                    
        elif type(uris) is str:
            if uris not in self.ontoURIs:
                print(f"Extracting ontology labels from {uris}.")
                self.__getLabels(uris)
            else:
                print(f"Labels already extracted from {uris}.")
        
        else:
            raise TypeError("uris must be a single uri string or a list of uri strings.")

        # Remove common short strings.
        self.labels.difference_update([
            # Numbers.
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            # One letter.
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
            "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
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
            "quality"
        ])

        print("Labels extracted.")
