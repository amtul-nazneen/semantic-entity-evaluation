
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_MLClassifier_Relation(allSentenceFeatures,allSentenceRelations):
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(allSentenceFeatures)
    Y = pd.np.array(allSentenceRelations)
    mnb = MultinomialNB()
    mnb.fit(X, Y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    return mnb,dv

def train_MLClassifier_Direction(allSentenceFeatures,allSentenceDirections):
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(allSentenceFeatures)
    Y = pd.np.array(allSentenceDirections)
    mnb = MultinomialNB()
    mnb.fit(X, Y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    return mnb,dv

def predict_MLClassifier_Direction(trainedDirectionModel,dictVectorizer,input):
    predictedDirection = trainedDirectionModel.predict(dictVectorizer.transform(input))
    return predictedDirection[0]


def predict_MLClassifier_Relation(trainedRelationModel,dictVectorizer,input):
    predictedRelation = trainedRelationModel.predict(dictVectorizer.transform(input))
    return predictedRelation[0]