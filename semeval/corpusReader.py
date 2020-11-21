import re
from semeval import preprocessor
from semeval.utils import *

def corpusReader():
    relationMap = preprocessor.getPreProcessedRelationMap()
    with open(TRAINING_FILE_NAME, 'r') as file:
        text = file.read()
        # should take these inputs from amtul if dict is the structure
        # processedParaList = [[0] * no_of_sentences] * no_of_features
        processedParaList = []
        paragraphs = [s for s in text.split('\n\n') if s]
        for para in paragraphs:
            feature_array = []
            e_1 = ""
            e_2 = ""
            relation_direction = 0
            paragraph_number = para.split("    \"")[0].replace("\n", "")
            quoted = re.compile('"[\d\D\w\W\s]+\n')
            onlyquoted = quoted.findall(para)
            if (onlyquoted):
                for value in onlyquoted:
                    e1 = re.compile('<e1>(.*?)</e1>').search(value)
                    e_1 = e1.group(1)
                    # processedParaList[paragraph_number][1]=e_1
                    e2 = re.compile('<e2>(.*?)</e2>').search(value)
                    e_2 = e2.group(1)
                    # processedParaList[paragraph_number][2] = e_2
                sentences = para.split(value)
                # processedParaList[3]=sentences[1].strip().split("(", 0)
                rd = sentences[1].strip().split("(")
                relation_name = rd[0].strip()
                reld = rd[1].replace(")", "")
                if (reld == "e1,e2"):
                    relation_direction = 0
                else:
                    relation_direction = 1
                # processedParaList[4] = relation_direction
                preprocessed_sent = value.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>",
                                                                                                               "")
                # textsentences[paragraph_number] = preprocessed_sent
                # processedParaList[0] = preprocessed_sent
                feature_array.append(preprocessed_sent)
                feature_array.append(e_1)
                feature_array.append(e_2)
                feature_array.append(relationMap.get(relation_name))
                feature_array.append(relation_direction)
                printConsole(feature_array)
                processedParaList.append(feature_array)

    return processedParaList