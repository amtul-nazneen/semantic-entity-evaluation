from semeval.common.utils import printConsole
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# TODO: Complete computations
def computePredictionScores(allSentenceExpectedRelations, allSentenceExpectedDirections,
                            allSentencePredictedRelations, allSentencePredictedDirections, semanticRelationMap):
    printConsole("Beginning computation")
    printConsole("Expected Relations: ")
    printConsole(allSentenceExpectedRelations)
    printConsole("Predicted Relations: ")
    printConsole(allSentencePredictedRelations)
    printConsole("Expected Directions: ")
    printConsole(allSentenceExpectedDirections)
    printConsole("Predicted Directions: ")
    printConsole(allSentencePredictedDirections)
    # calculating the metrics part-4
    acc_score = accuracy_score(allSentenceExpectedRelations, allSentencePredictedRelations)
    printConsole("Accuracy: ")
    printConsole(acc_score)
    printConsole("Precision, Recall and FScore : ")
    printConsole(
        precision_recall_fscore_support(allSentenceExpectedRelations,
                                        allSentencePredictedRelations, average='macro'))
    # calculating fscore and stats per label
    rel_labels = semanticRelationMap.keys()
    printConsole("Labels: ")
    printConsole(rel_labels)
    #rel_labels= ['Other', 'Content-Container', 'Entity-Origin', 'Message-Topic', 'Entity-Destination', 'Instrument-Agency']
    rel_labels = [1,2,3,4,5,6]
    printConsole(rel_labels)
    printConsole(precision_recall_fscore_support(allSentenceExpectedRelations,
                                                 allSentencePredictedRelations, average=None,
                                                 labels=rel_labels))