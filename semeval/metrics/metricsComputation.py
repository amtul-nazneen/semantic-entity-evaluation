from semeval.common.utils import printConsole
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def computePredictionScores(allSentenceExpectedRelations, allSentenceExpectedDirections,
                            allSentencePredictedRelations, allSentencePredictedDirections, semanticRelationMap):
    printConsole("METRIC COMPUTATION: Beginning computation")
    printConsole("METRIC COMPUTATION: Expected Relations from Training: ")
    printConsole(allSentenceExpectedRelations)
    printConsole("METRIC COMPUTATION: Predicted Relations from Training: ")
    printConsole(allSentencePredictedRelations)
    printConsole("METRIC COMPUTATION: Expected Directions from Training: ")
    printConsole(allSentenceExpectedDirections)
    printConsole("METRIC COMPUTATION: Predicted Directions from Training: ")
    printConsole(allSentencePredictedDirections)
    # calculating the metrics part-4
    acc_score = accuracy_score(allSentenceExpectedRelations, allSentencePredictedRelations)
    printConsole("######### METRICS under SETTING #1: Relation ######### ")
    printConsole("METRICS: Accuracy:")
    printConsole(acc_score)
    printConsole("METRICS: Overall - Precision, Recall, FScore:")
    printConsole(
        precision_recall_fscore_support(allSentenceExpectedRelations,
                                        allSentencePredictedRelations, average='macro'))
    # calculating fscore and stats per label
    semanticRelationMapValues = semanticRelationMap.values()
    relationLabels = []
    for relation in semanticRelationMapValues:
        relationLabels.append(relation)
    printConsole("METRICS: Unique Relation Labels: ")
    printConsole(relationLabels)
    printConsole("METRICS: Per Label - Precision, Recall, FScore")
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
    printConsole("######### METRICS under SETTING #2: Relation and Direction ######### ")
    printConsole("METRICS: Accuracy: ")
    printConsole(acc_score)
    printConsole("METRICS: Precision, Recall, FScore Per label: ")
    printConsole(
        precision_recall_fscore_support(expected_club,
                                        pred_club, average='macro'))