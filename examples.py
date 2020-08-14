#!/usr/bin/env python3

from .onto_labels import OntoLabels
# from .buildcorpus import makeCorpus

labels = OntoLabels()

ontoURIs = ["http://www.bioassayontology.org/bao/bao_complete.owl#",
            "http://purl.obolibrary.org/obo/po.owl#"]

labels.addOntoLabels(ontoURIs)



