#TASK4 - ML Model to classify relation
#Input - sentenceToRelationArray from deepNLPPipeline
#Output - trainedMLModelRelation - the trained model
def train_MLClassifier_Relation(sentenceToRelationArray):
    """
    inputFeatures, outputResults = train_MLClassifier_Helper(sentenceToRelationArray)
    run the classification using i/p and o/p
    """

#TASK4 - ML Model to classify relation
#Input - sentenceToDirectionArray from deepNLPPipeline
#Output - trainedMLModelDirection - the trained model
def train_MLClassifier_Direction(sentenceToDirectionArray):
    """
    inputFeatures, outputResults = train_MLClassifier_Helper(sentenceToDirectionArray)
    run the classification using i/p and o/p
    """

#TASK4: Helper method (1)
#Input: sentenceToRelationArray or sentenceToDirectionArray
#Output: inputFeatures, outputResults - separate arrays
def train_MLClassifier_Helper(sentenceToArray):
    """
    convert dict to two separate arrays as input and output
    """