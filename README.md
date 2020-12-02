# Semantic Entity Relation Evaluation
We are given a TAC KBP dataset using which we try to build a model to  predict relations and direction of relations between pre-tagged entities. After the training phase , a similar test set is provided to the built model, and we predict the relations and directions between its entities.  Accuracy , recall , macro precision and F-scores under settings relation-only and relation-direction are calculated and reported below. And also, given a test sentence, we load the same model and make relation, direction  prediction between the entities. The model we trained for the purpose is probabilistic.
### Getting Started
These instructions will get you a copy of the application up and running on your local machine for development and testing purposes.
### Softwares/SDKs
#### Required Tools and Softwares
    Python 3.7
    Stanford Core NLP
    SkLearn Naive Bayes
    Sklearn.feature_extraction DictVectorizer
    SpaCY
    NLTK Corpus-Wordnet
    Sklearn metrics
    _pickle
 

### End to End Setup Instructions
All the other libraries are imported using pip install

### Running stanford NLP server
``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000``

### Running the program
Download all the source code into a folder.
run the file ``main.py``
After model trains we can see the output containing metrics like accuracy ,precision, recall and fscore.

### Testing a given sentence
Test sentence can be tested at the end of training after all pickle files are generated.

Input Sentence:``The <e1>fortress</e1> has four <e2>towers</e2> corresponding to the cardinal points.``
2020-11-29 21:05:00 :  Entities: fortress:towers
2020-11-29 21:05:00 :  Modified Sentence: The fortress has four towers corresponding to the cardinal points.
2020-11-29 21:05:00 :  Modified entities: fortress:towers
2020-11-29 21:05:00 :   ################### SEMANTIC FEATURES #######################
2020-11-29 21:05:00 :  >> Extracted Tokens:
2020-11-29 21:05:00 :  ['The', 'fortress', 'has', 'four', 'towers', 'corresponding', 'to', 'the', 'cardinal', 'points']
2020-11-29 21:05:00 :  >> Extracted Dependency Parsing Tokens as Features:
2020-11-29 21:05:00 :  ['fortress', 'has', 'towers']
2020-11-29 21:05:00 :  >> Padded Tokens for Consistent Length:
2020-11-29 21:05:00 :  ['fortress', 'has', 'towers', '$', '$', '$', '$', '$', '$', '$', '$']
2020-11-29 21:05:00 :  >> Extracted Lemma as Features:
2020-11-29 21:05:00 :  {​​​​'lemma1': 'fortress', 'lemma2': 'have', 'lemma3': 'tower', 'lemma4': '$', 'lemma5': '$', 'lemma6': '$', 'lemma7': '$', 'lemma8': '$', 'lemma9': '$', 'lemma10': '$', 'lemma11': '$'}​​​​
2020-11-29 21:05:00 :  >> Extracted POS as Features:
2020-11-29 21:05:00 :  {​​​​'pos1': 'NN', 'pos2': 'VBZ', 'pos3': 'NNS', 'pos4': '$', 'pos5': '$', 'pos6': '$', 'pos7': '$', 'pos8': '$', 'pos9': '$', 'pos10': '$', 'pos11': '$'}​​​​
2020-11-29 21:05:00 :  >> Extracted NER as Features:
2020-11-29 21:05:00 :  {​​​​'entity1': 'OTHER', 'entity2': 'OTHER'}​​​​
2020-11-29 21:05:00 :  >> Extracted WordNet as Features:
2020-11-29 21:05:00 :  {​​​​'hyper1': 'defensive_structure', 'hyper2': 'structure', 'hypo1': 'alcazar', 'hypo2': 'barbican', 'holo1': '$', 'holo2': '$', 'mero1': 'battlement', 'mero2': 'helm'}​​​​
2020-11-29 21:05:00 :   ################### SEMANTIC FEATURES #######################
2020-11-29 21:05:00 :  >> NLP Pipeline Output: All-Sentence-Features
2020-11-29 21:05:00 :  {​​​​'lemma1': 'fortress', 'lemma2': 'have', 'lemma3': 'tower', 'lemma4': '$', 'lemma5': '$', 'lemma6': '$', 'lemma7': '$', 'lemma8': '$', 'lemma9': '$', 'lemma10': '$', 'lemma11': '$', 'pos1': 'NN', 'pos2': 'VBZ', 'pos3': 'NNS', 'pos4': '$', 'pos5': '$', 'pos6': '$', 'pos7': '$', 'pos8': '$', 'pos9': '$', 'pos10': '$', 'pos11': '$', 'entity1': 'OTHER', 'entity2': 'OTHER', 'hyper1': 'defensive_structure', 'hyper2': 'structure', 'hypo1': 'alcazar', 'hypo2': 'barbican', 'holo1': '$', 'holo2': '$', 'mero1': 'battlement', 'mero2': 'helm'}​​​​
2020-11-29 21:05:00 :  Beginning Prediction
2020-11-29 21:05:00 :  Loading Saved Model from Disk..
2020-11-29 21:05:00 :  >> Predicted Relation: Component-Whole
2020-11-29 21:05:00 :  >> Predicated Direction: (e2,e1)
2020-11-29 21:05:00 :  >> Total Prediction Time: 0.0009970664978027344 seconds
 
#### Files generated after the run
``consolelogFile`` - has the log of the program with metrics reported at the end
``input_features.txt`` - lists the input training file features like lemmas, tokens, pos tags, shortest depedency parsing , NER and wordnet features
``ml_model_classifier_direction.pkl`` - model created to predict direction of relation between entities
``ml_model_classifier_relation.pkl``- model created to predict relation of relation between entities
``vectorizer_direction.pickle``- dict vectorizer created to predict direction of relation between entities
``vectorizer_relation.pickle``- dict vectorizer created to predict relation of relation between entities
 
### Authors
    Amtul Nazneen - axn180041
    Hemanjeni Kundem - hxk180032
