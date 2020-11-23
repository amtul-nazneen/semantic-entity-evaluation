from semeval import preprocessor, featureExtractor, mlClassifier,corpusReader
from semeval.utils import *
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

def deepNLPPipeline(processedParaList,MAX_SENTENCE_LENGTH):
    printConsole("In deepNLPPipeline")
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
        paddedTokenArray = featureExtractor.padTokenArray(tokenArray,MAX_SENTENCE_LENGTH)
        lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
        POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
        nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
        wordNetArray = featureExtractor.extractWordNet_Features(paddedTokenArray)
        parsingArray = featureExtractor.extractParsing_Features(sentence,entity1,entity2)
        # printConsole("All Lemmas: ")
        # printConsole(lemmaArray)
        # printConsole("All POS: ")
        # printConsole(POSArray)
        # printConsole("All NER: ")
        # printConsole(nerArray)
        # printConsole("All WordNet: ")
        # printConsole(wordNetArray)
        # printConsole("All Parsing Array: ")
        # printConsole(parsingArray)
        allMergedFeatures = mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)
        printConsole("All Merged Features For Sentence:")
        printConsole(allMergedFeatures)
        allSentenceFeatures.append(allMergedFeatures)
        allSentenceDirections.append(direction)
        allSentenceRelations.append(relation)
    printConsole("Returning: allSentenceFeatures, allSentenceRelations, allSentenceDirections")
    printConsole(allSentenceFeatures)
    printConsole(allSentenceRelations)
    printConsole(allSentenceDirections)
    return allSentenceFeatures, allSentenceRelations, allSentenceDirections


def mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray):
    allMergedFeatures = lemmaArray.copy()
    allMergedFeatures.update(POSArray)
    allMergedFeatures.update(nerArray)
    allMergedFeatures.update(wordNetArray)
    return allMergedFeatures
