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

    # calculating the metrics part-4 b direction
    # clubbing both arrays relation, direction (expected)

    exp_relations = allSentenceExpectedRelations
    exp_directions = allSentenceExpectedDirections
    expected_club = []

    for i in exp_relations:
        for j in exp_directions:
            expected_club.append(i + j)
        exp_relations.remove(i)
    print(expected_club)

    # clubbing both arrays relation, direction (predicted)

    pred_relations = allSentencePredictedRelations
    pred_directions = allSentencePredictedDirections
    pred_club = []

    for i in pred_relations:
        for j in pred_directions:
            pred_club.append(i + j)
        pred_relations.remove(i)
    print(pred_club)

    acc_score = accuracy_score(expected_club, pred_club)
    printConsole("TESTING Accuracy : both relation and direction")
    printConsole(acc_score)
    printConsole("TESTING: Precision, Recall and FScore Per label: ")
    printConsole(
        precision_recall_fscore_support(expected_club,
                                        pred_club, average='macro'))