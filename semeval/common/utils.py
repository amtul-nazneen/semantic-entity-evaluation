import datetime

TRAINING_FILE_NAME = 'semeval_train_small.txt'
#TRAINING_FILE_NAME = '../data/semeval_train.txt'
#TRAINING_FILE_NAME = '../data/temp_test'
TESTING_FILE_NAME = 'semeval_test_small.txt'
#TESTING_FILE_NAME = '../data/semeval_test.txt'
WORD_NET_FEATURE_LENGTH = 1
PADDING_CHARACTER = '$'
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


def trim(input):
    if (input):
        inputAfterTrim = input.rstrip()
        if (inputAfterTrim):
            inputAfterTrim = input.lstrip()
    return inputAfterTrim

def printConsole(message):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"),": ", message)