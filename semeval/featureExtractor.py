import json
import string
from semeval import wordnetHelper
from semeval.utils import PADDING_CHARACTER, printConsole, NER_OTHER
import spacy
import networkx as nx


nlp_spacy = spacy.load('en_core_web_sm')
props = {
    'annotators': 'pos,lemma,depparse,ner',
    'pipelineLanguage': 'en',
    'outputFormat': 'json'
}
def extractTokens(sentence,nlp):
    sents = nlp.word_tokenize(sentence)
    sents = [s for s in sents if s not in string.punctuation]
    return sents


def extractLemma_Features(tokenArray,nlp):
    sentence = " ".join(tokenArray)
    parsed_str = nlp.annotate(sentence, properties=props)
    parsed_dict = json.loads(parsed_str)
    lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'lemma']

    # converting into dictionary
    res_lemma = {}
    # for key in tokenArray:
    for i in range(1, len(tokenArray) + 1):
        for value in lemma_list:
            res_lemma["lemma" + str(i)] = value
            lemma_list.remove(value)
            break

    return res_lemma


def extractPOS_Features(tokenArray,nlp):
    sentence = " ".join(tokenArray)
    parsed_str = nlp.annotate(sentence, properties=props)
    parsed_dict = json.loads(parsed_str)
    pos_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'pos']

    # converting into dictionary
    res_pos = {}
    # for key in tokenArray:
    # can also use - token_pos=dict(zip(sents_no_punct, pos_list))
    for i in range(1, len(tokenArray) + 1):
        for value in pos_list:
            res_pos["pos" + str(i)] = value
            pos_list.remove(value)
            break

    return res_pos

def extractNER_Features(tokenArray, entity1, entity2, nlp):
    sentence = " ".join(tokenArray)
    parsed_str = nlp.annotate(sentence, properties=props)
    parsed_dict = json.loads(parsed_str)
    ner_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'ner']
    res_ner = {}
    # for key in tokenArray:
    # can also use - token_pos=dict(zip(sents_no_punct, pos_list))
    ner_array = {}
    for key in tokenArray:
        for value in ner_list:
            res_ner[key] = value
            ner_list.remove(value)
            break

    e1_ner = res_ner[entity1]
    e2_ner = res_ner[entity2]
    if(e1_ner == 'O'):
        e1_ner =NER_OTHER
    if (e2_ner == 'O'):
        e2_ner = NER_OTHER
    ner_array["entity1"]= e1_ner
    ner_array["entity2"]= e2_ner
    return ner_array

def extractWordNet_Features(tokenArray):
    hypernyms = []
    hyponyms = []
    holonyms = []
    meronyms = []
    for token in tokenArray:
        hypernymsArray = wordnetHelper.extractWordNet_Hypernyms(token)
        for hypernym in hypernymsArray:
            hypernyms.append(hypernym)
        hyponymsArray = wordnetHelper.extractWordNet_Hyponyms(token)
        for hyponym in hyponymsArray:
            hyponyms.append(hyponym)
        meronymsArray = wordnetHelper.extractWordNet_Meronyms(token)
        for meronym in meronymsArray:
            meronyms.append(meronym)
        holonymsArray = wordnetHelper.extractWordNet_Holonyms(token)
        for holonym in holonymsArray:
            holonyms.append(holonym)
    # printConsole("hypernyms: ", hypernyms)
    # printConsole("hyponyms: ", hyponyms)
    # printConsole("holonyms: ", holonyms)
    # printConsole("meronyms: ", meronyms)
    allWordNetFeaturesDict =\
        wordnetHelper.extractWordNet_Features_Helper(len(tokenArray),hypernyms,hyponyms,holonyms,meronyms)
    return allWordNetFeaturesDict

def padTokenArray(tokenArray,MAX_SENTENCE_LENGTH):
    currentLength = len(tokenArray)
    difference = MAX_SENTENCE_LENGTH-currentLength
    while(difference>0):
        tokenArray.append(PADDING_CHARACTER)
        difference= difference-1
    printConsole("Updated Token Array:")
    printConsole(tokenArray)
    return tokenArray


def extractParsing_Features(sentence,entity1,entity2):
    doc = nlp_spacy(sentence)
    print('sentence:'.format(doc))  # Load spacy's dependency tree into a networkx graph
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
            graph = nx.Graph(edges)  # Get the length and path
    entity1 = entity1.lower()
    entity2 = entity2
    print(nx.shortest_path_length(graph, source=entity1, target=entity2))
    print(nx.shortest_path(graph, source=entity1, target=entity2))
    return nx.shortest_path(graph, source=entity1, target=entity2)

