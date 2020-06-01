#!/usr/bin/env python3

from getLabels.ontolabels import OntoLabels

labels = OntoLabels()

ontoURIs = ["http://www.bioassayontology.org/bao/bao_complete.owl#",
            "http://purl.obolibrary.org/obo/po.owl#"]

labels.addOntoLabels(ontoURIs)

print(labels.labels)

