#!/usr/bin/env python3

from .ontolabels import OntoLabels
# from .buildcorpus import makeCorpus

labels = OntoLabels()

ontoURIs = ["http://www.bioassayontology.org/bao/bao_complete.owl#",
            "http://purl.obolibrary.org/obo/po.owl#"]

labels.addOntoLabels(ontoURIs)



