
import re

from semeval import preprocessor, wordnetHelper, featureExtractor, mlClassifier
from semeval.utils import *
from stanfordcorenlp import StanfordCoreNLP
MAX_SENTENCE_LENGTH = preprocessor.getMaxSentenceLengthInTraining()
nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

#TODO: make a flow that trains and runs test sentences in a loop
# to do the testing
def main():
    printConsole("Training Begins")
    processedParaList = corpusReader()
    printConsole("Corpus Reader Completed")
    sentenceToRelationArray, sentenceToDirectionArray = deepNLPPipeline(processedParaList)
    trainedMLModelRelation = mlClassifier.train_MLClassifier_Relation(sentenceToRelationArray)
    trainedMLModelDirection = mlClassifier.train_MLClassifier_Relation(sentenceToDirectionArray)

#TASK1 - Read Input File
#Return : processedParaList - List of arrays Ex: ["Jack has a car", Jack, Car, 6,0]
def corpusReader(): #TODO- HEMA
    relationMap = preprocessor.getPreProcessedRelationMap()
    with open(TRAINING_FILE_NAME, 'r') as file:
        text = file.read()
        # should take these inputs from amtul if dict is the structure
        # processedParaList = [[0] * no_of_sentences] * no_of_features
        processedParaList = []
        paragraphs = [s for s in text.split('\n\n') if s]
        for para in paragraphs:
            feature_array = []
            e_1 = ""
            e_2 = ""
            relation_direction = 0
            paragraph_number = para.split("    \"")[0].replace("\n", "")
            quoted = re.compile('"[\d\D\w\W\s]+\n')
            onlyquoted = quoted.findall(para)
            if (onlyquoted):
                for value in onlyquoted:
                    e1 = re.compile('<e1>(.*?)</e1>').search(value)
                    e_1 = e1.group(1)
                    # processedParaList[paragraph_number][1]=e_1
                    e2 = re.compile('<e2>(.*?)</e2>').search(value)
                    e_2 = e2.group(1)
                    # processedParaList[paragraph_number][2] = e_2
                sentences = para.split(value)
                # processedParaList[3]=sentences[1].strip().split("(", 0)
                rd = sentences[1].strip().split("(")
                relation_name = rd[0].strip()
                reld = rd[1].replace(")", "")
                if (reld == "e1,e2"):
                    relation_direction = 0
                else:
                    relation_direction = 1
                # processedParaList[4] = relation_direction
                preprocessed_sent = value.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>",
                                                                                                               "")
                # textsentences[paragraph_number] = preprocessed_sent
                # processedParaList[0] = preprocessed_sent
                feature_array.append(preprocessed_sent)
                feature_array.append(e_1)
                feature_array.append(e_2)
                feature_array.append(relationMap.get(relation_name))
                feature_array.append(relation_direction)
                printConsole(feature_array)
                processedParaList.append(feature_array)

    return processedParaList


def deepNLPPipeline(processedParaList): #TODO - HEMA Old Code (Except Parsing - Later)
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



if __name__ == '__main__':
    main()
