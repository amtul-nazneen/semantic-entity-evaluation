import datetime

#TRAINING_FILE_NAME = '../data/semeval_train.txt'
TRAINING_FILE_NAME = 'semeval_train_small.txt'
WORD_NET_FEATURE_LENGTH = 3
PADDING_CHARACTER = '$$$'
HYPER_TAG = 'hyper'
HYPO_TAG = 'hypo'
HOLO_TAG = 'holo'
MERO_TAG = 'mero'
NER_OTHER = 'OTHER'

def trim(input):
    if (input):
        inputAfterTrim = input.rstrip()
        if (inputAfterTrim):
            inputAfterTrim = input.lstrip()
    return inputAfterTrim

def printConsole(message):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"),": ", message)