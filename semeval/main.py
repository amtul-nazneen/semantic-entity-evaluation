import time
import warnings

from semeval.common.utils import *
from semeval.preprocess import preprocessor
from semeval.workflow import semevalWorkflow


def main():
    warnings.simplefilter("ignore")
    printConsole("********** Team Hyderabadi Biryani: Semantic Evaluation In Named Entities **********")
    begin = time.time()
    semanticRelationMap,indexToRelationshipMap = preprocessor.getPreProcessedRelationMap()

    train(semanticRelationMap)

    test(semanticRelationMap)

    end = time.time()
    printConsole("Total Time Taken (Training + Testing): " + str(int(end-begin)) + " seconds" + "/" + str(int(int(end-begin)/60)) + " minutes")

    userInput(indexToRelationshipMap)


def train(semanticRelationMap):
    printConsole("********** Beginning Training Flow **********")
    semevalWorkflow.orchestrateTrainingFlow(TRAINING_FILE_NAME, semanticRelationMap)

def test(semanticRelationMap):
    semevalWorkflow.orchestrateTestingFlow(TESTING_FILE_NAME, semanticRelationMap)
    printConsole("********** Beginning Testing Flow **********")

def userInput(indexToRelationshipMap):
    printConsole("********** Beginning Manual User-Input Flow **********")
    semevalWorkflow.testInputSentence(indexToRelationshipMap)

if __name__ == '__main__':
    main()
