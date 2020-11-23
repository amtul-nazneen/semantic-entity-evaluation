from nltk.corpus import wordnet as wn

from semeval.common.utils import *


def extractWordNet_Features_Helper(totalTokens, hypernyms, hyponyms, holonyms, meronyms):
    upperLimit = WORD_NET_FEATURE_LENGTH * totalTokens
    if len(hypernyms) < upperLimit:
        diffToPad = upperLimit - len(hypernyms)
        while diffToPad > 0:
            hypernyms.append(PADDING_CHARACTER)
            diffToPad = diffToPad - 1
    if len(hyponyms) < upperLimit:
        diffToPad = upperLimit - len(hyponyms)
        while diffToPad > 0:
            hyponyms.append(PADDING_CHARACTER)
            diffToPad = diffToPad - 1
    if len(holonyms) < upperLimit:
        diffToPad = upperLimit - len(holonyms)
        while diffToPad > 0:
            holonyms.append(PADDING_CHARACTER)
            diffToPad = diffToPad - 1
    if len(meronyms) < upperLimit:
        diffToPad = upperLimit - len(meronyms)
        while diffToPad > 0:
            meronyms.append(PADDING_CHARACTER)
            diffToPad = diffToPad - 1
    hypernymDictValues = []
    hypernymDictKeys = []
    counter = 1
    for entry in hypernyms:
        hypernymDictKeys.append(HYPER_TAG + str(counter))
        hypernymDictValues.append(entry)
        counter = counter + 1
    hypernymDict = dict(zip(hypernymDictKeys, hypernymDictValues))

    hyponymDictValues = []
    hyponymDictKeys = []
    counter = 1
    for entry in hyponyms:
        hyponymDictKeys.append(HYPO_TAG + str(counter))
        hyponymDictValues.append(entry)
        counter = counter + 1
    hyponymDict = dict(zip(hyponymDictKeys, hyponymDictValues))

    holonymDictValues = []
    holonymDictKeys = []
    counter = 1
    for entry in holonyms:
        holonymDictKeys.append(HOLO_TAG + str(counter))
        holonymDictValues.append(entry)
        counter = counter + 1
    holonymDict = dict(zip(holonymDictKeys, holonymDictValues))

    meronymDictValues = []
    meronymDictKeys = []
    counter = 1
    for entry in meronyms:
        meronymDictKeys.append(MERO_TAG + str(counter))
        meronymDictValues.append(entry)
        counter = counter + 1
    meronymDict = dict(zip(meronymDictKeys, meronymDictValues))

    allWordNetFeaturesDict = {**hypernymDict, **hyponymDict, **holonymDict, **meronymDict}
    return allWordNetFeaturesDict


def extractWordNet_Hypernyms(token):
    hypernyms = []
    for ss in wn.synsets(token):
        if (len(hypernyms) < 2):
            for hyper in ss.hypernyms():
                for l in hyper.lemma_names():
                    if (len(hypernyms) < 2):
                        hypernyms.append(l)
    return hypernyms


def extractWordNet_Hyponyms(token):
    hyponyms = []
    for ss in wn.synsets(token):
        if (len(hyponyms) < 2):
            for hypo in ss.hyponyms():
                for l in hypo.lemma_names():
                    if (len(hyponyms) < 2):
                        hyponyms.append(l)
    return hyponyms


def extractWordNet_Meronyms(token):
    meronyms = []
    for ss in wn.synsets(token):
        if (len(meronyms) < 2):
            for mero in ss.part_meronyms():
                for l in mero.lemma_names():
                    if (len(meronyms) < 2):
                        meronyms.append(l)
    return meronyms


def extractWordNet_Holonyms(token):
    holonyms = []
    for ss in wn.synsets(token):
        if (len(holonyms) < 2):
            for holo in ss.part_holonyms():
                for l in holo.lemma_names():
                    if (len(holonyms) < 2):
                        holonyms.append(l)
    return holonyms
