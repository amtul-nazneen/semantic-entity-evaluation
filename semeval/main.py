import sys

#GLOBAL_VARIABLES
MAX_SENTENCE_LENGTH = 0
WORD_NET_FEATURE_LENGTH = 2

#TODO: make a flow that trains and runs test sentences in a loop
# to do the testing
def main():
    print("Beginning to Train")
    processedParaList = corpusReader()
    sentenceToRelationArray, sentenceToDirectionArray = deepNLPPipeline(processedParaList)
    trainedMLModelRelation = train_MLClassifier_Relation(sentenceToRelationArray)
    trainedMLModelDirection = train_MLClassifier_Relation(sentenceToDirectionArray)
    """
        run in a loop - where we take user input
        trainedMLModelRelation, trainedMLModelDirection can be used to predict
        """


# TASK - One-time method
#TODO: move to utils file
def preProcessingSentenceLength(): #TODO- AMTUL
    """
    1. Read the file, get the max length of the sentence - store in MAX_SENTENCE_LENGTH
    2. Count the unique relations
    """

# TASK0 - Extract Unique Relations and store their mappings
# Return: Map Ex: ["Entity-Destination":1..."Cause-Effect":6...]
def preProcessingRelationMap(): #TODO- AMTUL
    """
        1. Extract the unique relations from the file, store them in a map
        """

#TASK1 - Read Input File
#Return : processedParaList - List of arrays Ex: ["Jack has a car", Jack, Car, 6,0]
def corpusReader(): #TODO- HEMA
    relationMap = preProcessingRelationMap() #TODO-AMTUL
    """
    1. Begin read of the file
    3. Initialize a list [processedParaList] to hold arrays [each array has 5 entries]
    2. For each paragraph
        extract e1, e2 - store them
        extract relation - extract its value from 'relationMap' and store it
        extract direction - extract its boolean value and store it
        get rid of <e1> and <e2> and prepare the sentence
        Store the extracted elements' values in an array - [sentence, e1, e2, relation, direction]
        Ex: ["Jack has a car", Jack, Car, 6,0]
    """

#TASK2 - Extract Features
#Input: processedParaList from corpusReader() Ex: ["Jack has a car", Jack, Car, 6,0]
#Output: sentenceToRelationArray, sentenceToDirectionArray
def deepNLPPipeline(): #TODO - HEMA Old Code (Except WordNet - AMTUL) (Except Parsing - Later)
    """
        Initialize two arrays of dicts
            sentenceToRelationArray
            sentenceToDirectionArray
        For each entry in 'processedParaList'
            get entry0 in sentence
            get entry1 in entity1
            get entry2 in entity2
            tokenArray = extractTokens(sentence)
            lemmaArray = extractLemma_Features(tokenArray)
            POSArray = extractPOS_Features(tokenArray)
            nerArray = extractNER_Features(entity1, entity2)
            wordNetArray = extractWordNet_Features(tokenArray)
            parsingArray = extractParsing_Features(tokenArray) #TODO: how-to's
            #TODO: Constant Feature Length Processing
            Store all the (***Array results, direction) as an entry in sentenceToDirectionArray
            Ex: [{ "lemma1":"Jack", ........ "holo2":"blah"},0]
            Store all the (***Array results, relation) as an entry in sentenceToRelationArray
            Ex: [{ "lemma1":"Jack", ........ "holo2":"blah"},6]
        """

#TASK2 - helper method (1)
#Input - Sentence "Jack has a car"
#Output - Array of tokens {Jack,has,a,car}
def extractTokens(sentence):
    """
        Perform tokenization using Libraries
        """

#TASK2 - helper method (2)
#Input - Array of tokens {Jack,has,a,car}
#Output - Array of dicts {"lemma1":"Jack", "lemma2":"be" "lemma3":"a", "lemma4":"car"}
def extractLemma_Features(tokenArray):
    """
        for each token extract lemma, store in a dict
        #TODO: Constant feature length
        """

#TASK2 - helper method (3)
#Input - Array of tokens {Jack,has,a,car}
#Output - Array of dicts {"pos1":"NOUN", "pos2":"DET" "pos3":"ART", "pos4":"NOUN"}
def extractPOS_Features(tokenArray):
    """
        for each token extract POS, store in a dict
        #TODO: Constant feature length
        """

#TASK2 - helper method (4)
#Input - entity1, entity2 (Jack, car)
#Output - Array of dict {"ner1":"PERSON", "ner2":"THING"}
def extractNER_Features(entity1, entity2):
    """
        extract NER for both entities, store in a dict
        """
#TASK2 - helper method (5)
#Input - Array of tokens {Jack,has,a,car}
#Output - Array of dicts {"hyper1":"blah", ..... "holo2":"blah"}
def extractWordNet_Features(tokenArray):
    """
        Initialize wordNetArray
        for each token in tokenArray
            hypernymsArray = extractWordNet_Hypernyms(token)
            hyponymsArray = extractWordNet_Hyponyms(token)
            meronymsArray = extractWordNet_Meronyms(token)
            holonymsArray = extractWordNet_Holonyms(token)
            store all these in a flat array
            #TODO: carefully perform the indexing
        """


#TASK2 - helper method (6)
#Input: tokens {Jack}
#Output: Array of dicts {"hyper1":"blah", "hyper2":"blah"}
def extractWordNet_Hypernyms(token):
    """
        for given token extract hypernyms
        store only WORD_NET_FEATURE_LENGTH no. of hypernyms in the dict
        """
#TASK2 - helper method (7)
#Input: tokens {Jack}
#Output: Array of dicts {"hypo1":"blah", "hypo2":"blah"}
def extractWordNet_Hyponyms(token):
    """
        for given token extract hyponyms
        store only WORD_NET_FEATURE_LENGTH no. of hyponyms in the dict
        """

#TASK2 - helper method (8)
#Input: tokens {Jack}
#Output: Array of dicts {"mero1":"blah", "mero2":"blah"}
def extractWordNet_Meronyms(token):
    """
        for given token extract meronyms
        store only WORD_NET_FEATURE_LENGTH no. of meronyms in the dict
        """

#TASK2 - helper method (9)
#Input: tokens {Jack}
#Output: Array of dicts {"holo1":"blah", "holo2":"blah"}
def extractWordNet_Holonyms(token):
    """
        for given token extract holonyms
        store only WORD_NET_FEATURE_LENGTH no. of holonyms in the dict
        """

#TASK2 - helper method (9)
#Input:
#Output:
def extractParsing_Features(tokenArray):
    """
        #TODO: how-to's
        """


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


if __name__ == '__main__':
    main()
