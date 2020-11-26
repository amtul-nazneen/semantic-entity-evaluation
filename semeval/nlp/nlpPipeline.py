from semeval.nlp import featureExtractor
from semeval.common.utils import *
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

def deepNLPPipeline(processedParaList,MAX_SENTENCE_LENGTH,state):
    allSentenceFeatures = []
    allSentenceDirections = []
    allSentenceRelations = []
    for entry in processedParaList:
        sentence = entry[0]
        entity1Original = entry[1]
        entity2Original = entry[2]
        relation = entry[3]
        direction = entry[4]
        entity1,entity2 = concatenateAndChangeCase(entity1Original, entity2Original)
        #printConsole(state+"Original Sentence: " + sentence)
        #printConsole(state+"Original Entities: " + entity1Original + ":" + entity2Original)
        sentence = sentence.replace(entity1Original,entity1)
        sentence = sentence.replace(entity2Original, entity2)
        printConsole(state+"Sentence: " + sentence)
        printConsole(state+"Entities: " + entity1 + ":" + entity2)
        tokenizedArray = featureExtractor.extractTokens(sentence,nlp) #Invoking, but not used
        parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
        #printConsole(state+"Dependency Parsing Tokens: ")
        #printConsole(parsingArray)
        tokenArray = parsingArray
        paddedTokenArray = featureExtractor.padTokenArrayAndChangeCase(tokenArray, MAX_SENTENCE_LENGTH)
        lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
        POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
        nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
        wordnetInputOnlyEntity = []
        wordnetInputOnlyEntity.append(entity1)
        wordnetInputOnlyEntity.append(entity2)
        wordNetArray = featureExtractor.extractWordNet_Features(wordnetInputOnlyEntity)

        allMergedFeatures = mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)
        allSentenceFeatures.append(allMergedFeatures)
        allSentenceDirections.append(direction)
        allSentenceRelations.append(relation)
    printConsole(state+"NLP Pipeline Output: All-Sentence-Features")
    printConsole(allSentenceFeatures)
    printConsole(state+"NLP Pipeline Output: All-Sentence-Relations")
    printConsole(allSentenceRelations)
    printConsole(state+"NLP Pipeline Output: All-Sentence-Directions")
    printConsole(allSentenceDirections)
    return allSentenceFeatures, allSentenceRelations, allSentenceDirections

def mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray):
    allMergedFeatures = lemmaArray.copy()
    allMergedFeatures.update(POSArray)
    allMergedFeatures.update(nerArray)
    allMergedFeatures.update(wordNetArray)
    return allMergedFeatures


def getAllFeaturesForInputSentence(sentence,entity1Original, entity2Original,MAX_SENTENCE_LENGTH):
    entity1, entity2 = concatenateAndChangeCase(entity1Original, entity2Original)
    sentence = sentence.replace(entity1Original, entity1)
    sentence = sentence.replace(entity2Original, entity2)
    printConsole("Modified Sentence: " + sentence)
    printConsole("Modified entities: " + entity1 + ":" + entity2)

    tokenizedArray = featureExtractor.extractTokens(sentence, nlp)  # Invoking, but not used
    printConsole("Extracted Tokens:")
    printConsole(tokenizedArray)

    parsingArray = featureExtractor.extractParsing_Features(sentence, entity1, entity2)
    printConsole("Extracted Dependency Parsing Tokens as Features: ")
    printConsole(parsingArray)
    tokenArray = parsingArray

    paddedTokenArray = featureExtractor.padTokenArrayAndChangeCase(tokenArray, MAX_SENTENCE_LENGTH)
    printConsole("Padded Tokens for Consistent Length:")
    printConsole(paddedTokenArray)

    lemmaArray = featureExtractor.extractLemma_Features(paddedTokenArray, nlp)
    printConsole("Extracted Lemma as Features:")
    printConsole(lemmaArray)

    POSArray = featureExtractor.extractPOS_Features(paddedTokenArray, nlp)
    printConsole("Extracted POS as Features:")
    printConsole(POSArray)

    nerArray = featureExtractor.extractNER_Features(paddedTokenArray, entity1, entity2, nlp)
    printConsole("Extracted NER as Features:")
    printConsole(nerArray)

    wordnetInputOnlyEntity = []
    wordnetInputOnlyEntity.append(entity1)
    wordnetInputOnlyEntity.append(entity2)
    wordNetArray = featureExtractor.extractWordNet_Features(wordnetInputOnlyEntity)
    printConsole("Extracted WordNet as Features:")
    printConsole(wordNetArray)

    return mergingAllFeatures(lemmaArray,POSArray,nerArray,wordNetArray)


def concatenateAndChangeCase(entity1, entity2):
    entity1Concatenated = ""
    entity2Concatenated = ""
    words = entity1.strip().split()
    for word in words:
        entity1Concatenated = entity1Concatenated + word
        entity1Concatenated = entity1Concatenated.replace("-","")
    words = entity2.strip().split()
    for word in words:
        entity2Concatenated = entity2Concatenated + word
        entity2Concatenated = entity2Concatenated.replace("-", "")
    return entity1Concatenated.lower(), entity2Concatenated.lower()