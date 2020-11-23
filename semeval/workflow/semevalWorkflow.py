from pip._vendor.distlib.compat import raw_input

from semeval.classifier import mlClassifier
from semeval.metrics import metricsComputation
from semeval.nlp import nlpPipeline
from semeval.preprocess import preprocessor, corpusReader
from semeval.common.utils import *

MAX_SENTENCE_LENGTH = preprocessor.getMaxSentenceLengthInTraining()

def orchestrateTrainingFlow(fileName, semanticRelationMap):
    processedParaListTrain = corpusReader.readFile(fileName,semanticRelationMap)
    printConsole(">>>>>> Training Flow: Corpus Reader Completed")
    allSentenceFeatures, allSentenceRelations, allSentenceDirections = \
        nlpPipeline.deepNLPPipeline(processedParaListTrain, MAX_SENTENCE_LENGTH)
    printConsole(">>>>>> Training Flow: NLP Pipeline Completed")
    printConsole(">>>>>> Training Flow: Invoking Classifiers")
    trainedMLModelRelation, dictVectorRelation =\
        mlClassifier.train_MLClassifier_Relation(allSentenceFeatures, allSentenceRelations)
    printConsole("ML Learning Model for Relation Classification is Complete")
    trainedMLModelDirection, dictVectorDirection = \
        mlClassifier.train_MLClassifier_Direction(allSentenceFeatures, allSentenceDirections)
    printConsole("ML Learning Model for Direction Classification is Complete")
    return trainedMLModelRelation, dictVectorRelation, trainedMLModelDirection, dictVectorDirection

def orchestrateTestingFlow(fileName,semanticRelationMap,
                           trainedMLModelRelation, dictVectorRelation, trainedMLModelDirection, dictVectorDirection):
    processedParaListTest = corpusReader.readFile(fileName,semanticRelationMap)
    printConsole(">>>>>> Testing Flow: Corpus Reader Completed")
    allSentenceFeatures, allSentenceExpectedRelations, allSentenceExpectedDirections = \
        nlpPipeline.deepNLPPipeline(processedParaListTest, MAX_SENTENCE_LENGTH)
    printConsole(">>>>>> Testing Flow: NLP Pipeline Completed")
    allSentencePredictedRelations = []
    allSentencePredictedDirections = []
    printConsole(">>>>>> Testing Flow: Beginning Predictions for each sentence")
    for inputSentenceFeature in allSentenceFeatures:
        predictedRelation = mlClassifier.predict_MLClassifier_Relation\
            (trainedMLModelRelation,dictVectorRelation,inputSentenceFeature)
        predictedDirection = mlClassifier.predict_MLClassifier_Direction\
            (trainedMLModelDirection,dictVectorDirection,inputSentenceFeature)
        allSentencePredictedRelations.append(predictedRelation)
        allSentencePredictedDirections.append(predictedDirection)
    printConsole(">>>>>> Testing Flow: All Predictions Completed")
    printConsole(">>>>>> Testing Flow: Computing Metrics")
    metricsComputation.computePredictionScores(allSentenceExpectedRelations, allSentenceExpectedDirections,
                            allSentencePredictedRelations,allSentencePredictedDirections)


def testInputSentence(trainedMLModelRelation, dictVectorRelation,
                                           trainedMLModelDirection, dictVectorDirection,indexToRelationshipMap):
    loop = True
    printConsole(">>>>>> Manual User-Test Prediction: Enter the Sentence or " + INPUT_STOP_WORD + " to exit.")
    while loop:
        inputSentence = raw_input('Input Sentence:')
        if inputSentence == INPUT_STOP_WORD:
            printConsole("Exiting the Program")
            loop = False
        else:
            entity1 = raw_input('Entity_1:')
            entity2 = raw_input('Entity_2:')
            inputSentenceFeature = nlpPipeline.getAllFeaturesForInputSentence(inputSentence,entity1,entity2,MAX_SENTENCE_LENGTH)
            predictedRelation = mlClassifier.predict_MLClassifier_Relation \
                (trainedMLModelRelation, dictVectorRelation, inputSentenceFeature)
            predictedDirection = mlClassifier.predict_MLClassifier_Direction \
                (trainedMLModelDirection, dictVectorDirection, inputSentenceFeature)
            relation  = indexToRelationshipMap.get(predictedRelation[0])
            if(predictedDirection[0] ==E1_to_E2):
                direction = E1_E2
            else:
                direction = E2_E1
            printConsole("Predicted Relation: " + relation)
            printConsole("Predicated Direction: " + "("+ direction+")")


