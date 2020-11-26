import datetime
import logging

logging.basicConfig(level=logging.INFO, filename="consolelogFile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")



#TRAINING_FILE_NAME = '../data/smallSet/semeval_train_small.txt'
#TESTING_FILE_NAME = '../data/smallSet/semeval_test_small.txt'

TRAINING_FILE_NAME = '../data/semeval_train.txt'
TESTING_FILE_NAME = '../data/semeval_test.txt'

#TRAINING_FILE_NAME = 'semeval_train_small_metrics.txt'
#TESTING_FILE_NAME = 'semeval_test_small_metrics.txt'

TEST_STATE = "TESTING: "
TRAIN_STATE = "TRAINING: "

WORD_NET_FEATURE_LENGTH = 1
WORD_NET_FEATURE_UPPER_LIMIT=2
PADDING_CHARACTER = '$'
MAX_SENTENCE_LENGTH = 11

HYPER_TAG = 'hyper'
HYPO_TAG = 'hypo'
HOLO_TAG = 'holo'
MERO_TAG = 'mero'
NER_OTHER = 'OTHER'

INPUT_STOP_WORD = "EXIT"

E1_to_E2 = 0
E2_to_E1 = 1
E1_E2 = "e1,e2"
E2_E1 = "e2,e1"

RELATION_CLASSIFIER_TO_DISK = 'ml_model_classifier_relation.pkl'
DIRECTION_CLASSIFIER_TO_DISK = 'ml_model_classifier_direction.pkl'
RELATION_VECTORIZER_TO_DISK = "vectorizer_relation.pickle"
DIRECTION_VECTORIZER_TO_DISK = "vectorizer_direction.pickle"
WRITE_MODE = 'wb'
READ_MODE = 'rb'

def trim(input):
    if (input):
        inputAfterTrim = input.rstrip()
        if (inputAfterTrim):
            inputAfterTrim = input.lstrip()
    return inputAfterTrim

def printConsole(message):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"),": ", message)
    logging.info(str(now.strftime("%Y-%m-%d %H:%M:%S")) + ": " + str(message))