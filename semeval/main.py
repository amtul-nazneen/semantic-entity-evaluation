import warnings

from semeval.preprocess import preprocessor
from semeval.common.utils import *
from semeval.workflow import semevalWorkflow
import time

def main():
    printConsole("********** Team Hyderabadi Biryani: Semantic Evaluation In Named Entities **********")
    begin = time.time()
    warnings.simplefilter("ignore")
    semanticRelationMap,indexToRelationshipMap = preprocessor.getPreProcessedRelationMap()
    printConsole("********** Beginning Training Flow **********")
    #trainedMLModelRelation, dictVectorRelation, trainedMLModelDirection, dictVectorDirection = \
    semevalWorkflow.orchestrateTrainingFlow(TRAINING_FILE_NAME, semanticRelationMap)
    printConsole("********** Beginning Testing Flow **********")
    #semevalWorkflow.orchestrateTestingFlow(TESTING_FILE_NAME, semanticRelationMap,
    #                                      trainedMLModelRelation, dictVectorRelation,
    #                                     trainedMLModelDirection, dictVectorDirection)
    #semevalWorkflow.orchestrateTestingFlow(TESTING_FILE_NAME, semanticRelationMap)
    end = time.time()
    printConsole("Total Time Taken: " + str(int(end-begin)) + " seconds")
    printConsole("********** Beginning Manual User-Input Flow **********")
    #semevalWorkflow.testInputSentence(trainedMLModelRelation, dictVectorRelation,
    #                                  trainedMLModelDirection, dictVectorDirection, indexToRelationshipMap)


if __name__ == '__main__':
    main()
