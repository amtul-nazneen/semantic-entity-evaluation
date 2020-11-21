from semeval import preprocessor, featureExtractor, mlClassifier,corpusReader
from semeval.utils import *
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)
def deepNLPPipeline(processedParaList):
    printConsole("In deepNLPPipeline")
    sentenceToRelationArray = {}
    sentenceToDirectionArray = {}
    for entry in processedParaList:
        sentence = entry[0]
        entity1 = entry[1]
        entity2 = entry[2]
        tokenArray = featureExtractor.extractTokens(sentence, nlp)
        #paddedTokenArray = paddTokens (tokenArray) #TODO-AMTUL
        lemmaArray = featureExtractor.extractLemma_Features(tokenArray, nlp)
        POSArray = featureExtractor.extractPOS_Features(tokenArray, nlp)
        nerArray = featureExtractor.extractNER_Features(tokenArray, entity1, entity2, nlp)
        wordNetArray = featureExtractor.extractWordNet_Features(tokenArray)
        printConsole("All Lemmas: ")
        printConsole(lemmaArray)
        printConsole("All POS: ")
        printConsole(POSArray)
        printConsole("All NER: ")
        printConsole(nerArray)
        printConsole("All WordNet: ")
        printConsole(wordNetArray)
        # parsingArray = featureExtractor.extractParsing_Features(tokenArray)  # TODO: how-to's

