import re

from pip._vendor.distlib.compat import raw_input
from semeval.classifier import mlClassifier
from semeval.metrics import metricsComputation
from semeval.nlp import nlpPipeline
from semeval.preprocess import corpusReader
from semeval.common.utils import *
import _pickle as cPickle

#TODO: Preprocess max dependency path sentence length
#MAX_SENTENCE_LENGTH = preprocessor.getMaxSentenceLengthInTraining()

def orchestrateTrainingFlow(fileName, semanticRelationMap):
    processedParaListTrain = corpusReader.readFile(fileName,semanticRelationMap)
    printConsole(">>>>>> Training Flow: Corpus Reader Completed")
    printConsole(">>>>>> Training Flow: NLP Pipeline Beginning")
    allSentenceFeatures, allSentenceRelations, allSentenceDirections = \
        nlpPipeline.deepNLPPipeline(processedParaListTrain, MAX_SENTENCE_LENGTH)
    printConsole(">>>>>> Training Flow: NLP Pipeline Completed")
    printConsole(">>>>>> Training Flow: Invoking Classifiers")
    trainedMLModelRelation, dictVectorRelation =\
        mlClassifier.train_MLClassifier_Relation(allSentenceFeatures, allSentenceRelations)
    printConsole(">>>>>> Training Flow: ML Learning Model for Relation Classification is Complete")
    trainedMLModelDirection, dictVectorDirection = \
        mlClassifier.train_MLClassifier_Direction(allSentenceFeatures, allSentenceDirections)
    printConsole(">>>>>> Training Flow: ML Learning Model for Direction Classification is Complete")

    # save the classifier
    with open(RELATION_CLASSIFIER_TO_DISK, WRITE_MODE) as fid_rel:
        cPickle.dump(trainedMLModelRelation, fid_rel)
    cPickle.dump(dictVectorRelation, open(RELATION_VECTORIZER_TO_DISK, WRITE_MODE))
    with open(DIRECTION_CLASSIFIER_TO_DISK, WRITE_MODE) as fid_dir:
        cPickle.dump(trainedMLModelDirection, fid_dir)
    cPickle.dump(dictVectorDirection, open(DIRECTION_VECTORIZER_TO_DISK, WRITE_MODE))

    printConsole(">>>>>> Training model saving is Complete")

def orchestrateTestingFlow(fileName,semanticRelationMap ):
    processedParaListTest = corpusReader.readFile(fileName,semanticRelationMap)
    printConsole(">>>>>> Testing Flow: Corpus Reader Completed")
    printConsole(">>>>>> Testing Flow: NLP Pipeline Beginning")
    allSentenceFeatures, allSentenceExpectedRelations, allSentenceExpectedDirections = \
        nlpPipeline.deepNLPPipeline(processedParaListTest, MAX_SENTENCE_LENGTH)
    printConsole(">>>>>> Testing Flow: NLP Pipeline Completed")
    allSentencePredictedRelations = []
    allSentencePredictedDirections = []
    printConsole(">>>>>> Testing Flow: Beginning Predictions for each sentence")

    with open(RELATION_CLASSIFIER_TO_DISK, READ_MODE) as fid_rel:
        trainedMLModelRelation = cPickle.load(fid_rel)
    dictVectorRelation = cPickle.load(open(RELATION_VECTORIZER_TO_DISK, READ_MODE))

    with open(DIRECTION_CLASSIFIER_TO_DISK, 'rb') as fid_dir:
        trainedMLModelDirection = cPickle.load(fid_dir)
    dictVectorDirection = cPickle.load(open(DIRECTION_VECTORIZER_TO_DISK, READ_MODE))

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
                            allSentencePredictedRelations,allSentencePredictedDirections,semanticRelationMap)


def testInputSentence(indexToRelationshipMap):
    with open(RELATION_CLASSIFIER_TO_DISK, 'rb') as fid_rel:
        trainedMLModelRelation = cPickle.load(fid_rel)
    dictVectorRelation = cPickle.load(open(RELATION_VECTORIZER_TO_DISK, READ_MODE))

    with open(DIRECTION_CLASSIFIER_TO_DISK, 'rb') as fid_dir:
        trainedMLModelDirection = cPickle.load(fid_dir)
    dictVectorDirection = cPickle.load(open(DIRECTION_VECTORIZER_TO_DISK, READ_MODE))
    loop = True
    printConsole(">>>>>> Manual User-Test Prediction: Enter the Sentence in below format or " + INPUT_STOP_WORD + " to exit.")
    printConsole("Jack and Jill went to the <e1>hill</e1> to fetch a pail of <e2>water</e2>.")
    while loop:
        inputSentence = raw_input('Input Sentence:')
        if inputSentence == INPUT_STOP_WORD:
            printConsole("Exiting the Program")
            loop = False
        else:
            e1 = re.compile('<e1>(.*?)</e1>').search(inputSentence)
            entity1 = e1.group(1)
            e2 = re.compile('<e2>(.*?)</e2>').search(inputSentence)
            entity2 = e2.group(1)
            printConsole("Entities: " + entity1 + ":" + entity2)
            inputSentence = inputSentence.replace("<e1>", "") \
                .replace("</e1>", "").replace("<e2>", "") \
                .replace("</e2>", "")
            inputSentenceFeature = nlpPipeline.getAllFeaturesForInputSentence(inputSentence,entity1,entity2,MAX_SENTENCE_LENGTH)
            printConsole("NLP Pipeline Output: All-Sentence-Features")
            printConsole(inputSentenceFeature)
            predictedRelation = mlClassifier.predict_MLClassifier_Relation \
                (trainedMLModelRelation, dictVectorRelation, inputSentenceFeature)
            predictedDirection = mlClassifier.predict_MLClassifier_Direction \
                (trainedMLModelDirection, dictVectorDirection, inputSentenceFeature)
            relation  = indexToRelationshipMap.get(predictedRelation)
            if(predictedDirection ==E1_to_E2):
                direction = E1_E2
            else:
                direction = E2_E1
            printConsole("Predicted Relation: " + relation)
            printConsole("Predicated Direction: " + "("+ direction+")")


