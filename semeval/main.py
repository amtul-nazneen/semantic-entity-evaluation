import time
import warnings

from semeval.common.utils import *
from semeval.preprocess import preprocessor
from semeval.workflow import semevalWorkflow


def main():
    printConsole("********** Team Hyderabadi Biryani: Semantic Evaluation In Named Entities **********")
    begin = time.time()
    warnings.simplefilter("ignore")

    #PRE-PROCESSED RELATION MAP
    semanticRelationMap,indexToRelationshipMap = preprocessor.getPreProcessedRelationMap()
    printConsole("********** Beginning Training Flow **********")

    # TRAINING PHASE
    semevalWorkflow.orchestrateTrainingFlow(TRAINING_FILE_NAME, semanticRelationMap)
    printConsole("********** Beginning Testing Flow **********")

    # TESTING PHASE
    semevalWorkflow.orchestrateTestingFlow(TESTING_FILE_NAME, semanticRelationMap)
    end = time.time()

    printConsole("Total Time Taken: " + str(int(end-begin)) + " seconds")

    # USER INPUT TESTING PHASE
    printConsole("********** Beginning Manual User-Input Flow **********")
    semevalWorkflow.testInputSentence(indexToRelationshipMap)


if __name__ == '__main__':
    main()
