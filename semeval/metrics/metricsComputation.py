from semeval.common.utils import printConsole
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# TODO: Complete computations
def computePredictionScores(allSentenceExpectedRelations, allSentenceExpectedDirections,
                            allSentencePredictedRelations, allSentencePredictedDirections, semanticRelationMap):
    printConsole("TRAINING: Beginning computation")
    printConsole("TRAINING: Expected Relations: ")
    printConsole(allSentenceExpectedRelations)
    printConsole("TRAINING: Predicted Relations: ")
    printConsole(allSentencePredictedRelations)
    printConsole("TRAINING: Expected Directions: ")
    printConsole(allSentenceExpectedDirections)
    printConsole("TRAINING: Predicted Directions: ")
    printConsole(allSentencePredictedDirections)
    # calculating the metrics part-4
    acc_score = accuracy_score(allSentenceExpectedRelations, allSentencePredictedRelations)
    printConsole("TRAINING: Accuracy: ")
    printConsole(acc_score)
    printConsole("TRAINING: Precision, Recall and FScore : ")
    printConsole(
        precision_recall_fscore_support(allSentenceExpectedRelations,
                                        allSentencePredictedRelations, average='macro'))
    # calculating fscore and stats per label
    semanticRelationMapValues = semanticRelationMap.values()
    relationLabels = []
    for relation in semanticRelationMapValues:
        relationLabels.append(relation)
    printConsole("TRAINING: Relation Labels: ")
    printConsole(relationLabels)
    printConsole("TRAINING: Precision, Recall and FScore Per Label: ")
    printConsole(precision_recall_fscore_support(allSentenceExpectedRelations,
                                                 allSentencePredictedRelations, average=None,
                                                 labels=relationLabels))