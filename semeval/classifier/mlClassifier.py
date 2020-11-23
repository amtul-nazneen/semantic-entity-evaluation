
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


# --> !!Not using below code: Storing for future may be
# def dict_vectorizer(allSentenceFeatures):
#     vectorizer = DictVectorizer(sparse=True)
#     feature_dict_vector = vectorizer.fit_transform(allSentenceFeatures).toarray()
#     printConsole("Printing Feature Dict Vector")
#     printConsole(feature_dict_vector)
#     return feature_dict_vector