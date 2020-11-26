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
    semanticRelationMapValues = semanticRelationMap.values()
    relationLabels = []
    for relation in semanticRelationMapValues:
        relationLabels.append(relation)
    printConsole("Relation Labels: ")
    printConsole(relationLabels)
    printConsole(precision_recall_fscore_support(allSentenceExpectedRelations,
                                                 allSentencePredictedRelations, average=None,
                                                 labels=relationLabels))