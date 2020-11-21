from semeval import preprocessor, mlClassifier,corpusReader,nlpPipeline
from semeval.utils import *

MAX_SENTENCE_LENGTH = preprocessor.getMaxSentenceLengthInTraining()

def main():
    printConsole("Training Begins")
    processedParaList = corpusReader.corpusReader()
    printConsole("Corpus Reader Completed")
    sentenceToRelationArray, sentenceToDirectionArray = nlpPipeline.deepNLPPipeline(processedParaList)
    printConsole("Training Model for Relation")
    trainedMLModelRelation = mlClassifier.train_MLClassifier_Relation(sentenceToRelationArray)
    printConsole("Training Model for Direction")
    trainedMLModelDirection = mlClassifier.train_MLClassifier_Relation(sentenceToDirectionArray)
    printConsole("Learned Model is Ready")
    #TODO- Code for prediction

if __name__ == '__main__':
    main()
