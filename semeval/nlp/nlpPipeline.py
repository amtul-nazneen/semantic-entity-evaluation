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
        tokenArray = featureExtractor.extractTokens(sentence, nlp)
        paddedTokenArray = featureExtractor.padTokenArray(tokenArray, MAX_SENTENCE_LENGTH)
        lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
        POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
        nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
        wordNetArray = featureExtractor.extractWordNet_Features(paddedTokenArray)
        parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
        allMergedFeatures = mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)
        # printConsole("All Merged Features For Sentence:")
        # printConsole(allMergedFeatures)
        allSentenceFeatures.append(allMergedFeatures)
        allSentenceDirections.append(direction)
        allSentenceRelations.append(relation)
    printConsole("NLP Pipeline: All-Sentence-Features")
    printConsole(allSentenceFeatures)
    printConsole("NLP Pipeline: All-Sentence-Relations")
    printConsole(allSentenceRelations)
    printConsole("NLP Pipeline: All-Sentence-Directions")
    printConsole(allSentenceDirections)
    return allSentenceFeatures, allSentenceRelations, allSentenceDirections


def mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray):
    allMergedFeatures = lemmaArray.copy()
    allMergedFeatures.update(POSArray)
    allMergedFeatures.update(nerArray)
    allMergedFeatures.update(wordNetArray)
    return allMergedFeatures


def getAllFeaturesForInputSentence(sentence,entity1, entity2,MAX_SENTENCE_LENGTH):
    tokenArray = featureExtractor.extractTokens(sentence, nlp)
    paddedTokenArray = featureExtractor.padTokenArray(tokenArray, MAX_SENTENCE_LENGTH)
    lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
    POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
    nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
    wordNetArray = featureExtractor.extractWordNet_Features(paddedTokenArray)
    parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
    return mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)