from semeval.nlp import featureExtractor
from semeval.common.utils import *
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

def deepNLPPipeline(processedParaList,MAX_SENTENCE_LENGTH):
    allSentenceFeatures = []
    allSentenceDirections = []
    allSentenceRelations = []
    for entry in processedParaList:
        sentence = entry[0]
        entity1 = entry[1]
        entity2 = entry[2]
        relation = entry[3]
        direction = entry[4]
        parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
        printConsole("Dependency Parsing Tokens for given sentence: ")
        printConsole(parsingArray)
        tokenArray = parsingArray
        paddedTokenArray = featureExtractor.padTokenArray(tokenArray, MAX_SENTENCE_LENGTH)
        lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
        POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
        nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
        wordNetArray = featureExtractor.extractWordNet_Features(paddedTokenArray)

        allMergedFeatures = mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)
        allSentenceFeatures.append(allMergedFeatures)
        allSentenceDirections.append(direction)
        allSentenceRelations.append(relation)
    printConsole("NLP Pipeline Output: All-Sentence-Features")
    printConsole(allSentenceFeatures)
    printConsole("NLP Pipeline Output: All-Sentence-Relations")
    printConsole(allSentenceRelations)
    printConsole("NLP Pipeline Output: All-Sentence-Directions")
    printConsole(allSentenceDirections)
    return allSentenceFeatures, allSentenceRelations, allSentenceDirections

def mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray):
    allMergedFeatures = lemmaArray.copy()
    allMergedFeatures.update(POSArray)
    allMergedFeatures.update(nerArray)
    allMergedFeatures.update(wordNetArray)
    return allMergedFeatures


def getAllFeaturesForInputSentence(sentence,entity1, entity2,MAX_SENTENCE_LENGTH):
    parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
    printConsole("Dependency Parsing Tokens for input sentence: ")
    printConsole(parsingArray)
    tokenArray = parsingArray
    paddedTokenArray = featureExtractor.padTokenArray(tokenArray, MAX_SENTENCE_LENGTH)
    lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
    POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
    nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
    wordNetArray = featureExtractor.extractWordNet_Features(paddedTokenArray)

    return mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)