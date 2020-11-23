from semeval import preprocessor, mlClassifier,corpusReader,nlpPipeline
from semeval.utils import *

MAX_SENTENCE_LENGTH = preprocessor.getMaxSentenceLengthInTraining()

def main():
    printConsole("Training Begins")
    processedParaList = corpusReader.corpusReader()
    printConsole("Corpus Reader Completed")
    allSentenceFeatures, allSentenceRelations, allSentenceDirections = \
    nlpPipeline.deepNLPPipeline(processedParaList,MAX_SENTENCE_LENGTH)
    printConsole("Training Model for Relation")
    trainedMLModelRelation = mlClassifier.train_MLClassifier_Relation(allSentenceFeatures,allSentenceRelations)
    printConsole("Training Model for Direction")
    trainedMLModelDirection = mlClassifier.train_MLClassifier_Direction(allSentenceFeatures,allSentenceDirections)
    printConsole("Learned Model is Ready")
    #TODO- Code for prediction

if __name__ == '__main__':
    main()
