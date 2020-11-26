from pip._vendor.distlib.compat import raw_input
from sklearn.feature_extraction import DictVectorizer

from semeval.classifier import mlClassifier
from semeval.metrics import metricsComputation
from semeval.nlp import nlpPipeline
from semeval.preprocess import preprocessor, corpusReader
from semeval.common.utils import *
import _pickle as cPickle

#TODO: Preprocess max dependency path sentence length
MAX_SENTENCE_LENGTH = 11#preprocessor.getMaxSentenceLengthInTraining()

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
    with open('ml_model_classifier_relation.pkl', 'wb') as fid_rel:
        cPickle.dump(trainedMLModelRelation, fid_rel)
    cPickle.dump(dictVectorRelation, open("vectorizer_relation.pickle", "wb"))
    with open('ml_model_classifier_direction.pkl', 'wb') as fid_dir:
        cPickle.dump(trainedMLModelDirection, fid_dir)
    cPickle.dump(dictVectorDirection, open("vectorizer_firection.pickle", "wb"))

    printConsole(">>>>>> Training model saving is Complete")
        #return trainedMLModelRelation, dictVectorRelation, trainedMLModelDirection, dictVectorDirection

#def orchestrateTestingFlow(fileName,semanticRelationMap,
#                          trainedMLModelRelation, dictVectorRelation, trainedMLModelDirection, dictVectorDirection):
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

    with open('ml_model_classifier_relation.pkl', 'rb') as fid_rel:
        trainedMLModelRelation = cPickle.load(fid_rel)

    dictVectorRelation = cPickle.load(open("vectorizer_relation.pickle", "rb"))

    with open('ml_model_classifier_direction.pkl', 'rb') as fid_dir:
        trainedMLModelDirection = cPickle.load(fid_dir)

    dictVectorDirection = cPickle.load(open("vectorizer_firection.pickle", "rb"))

    #dictVectorRelation= DictVectorizer(sparse=True)
    #dictVectorDirection = DictVectorizer(sparse=True)

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


