#!/usr/bin/env python3


import owlready2

# uri = "http://www.bioassayontology.org/bao/bao_complete.owl#"

class OntoLabels:
    
    def __init__(self):
        
        self.labels = set()
        self.ontoURIs = set()

    def __getLabels(self, uri):
        self.ontoURIs.add(uri)
        onto = owlready2.get_ontology(uri).load()
        for c in onto.classes():
            try:
                self.labels.add(c.label[0])
            except IndexError:
                pass
            
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
                self.__getLabels(uri)
            else:
                print(f"Labels already extracted from {uris}.")
        
        else:
            raise TypeError("uris must be a single uri string or a list of uri strings.")
                
