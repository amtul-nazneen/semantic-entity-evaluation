import re
from semeval.common.utils import *

def readFile(semevalFile,semanticRelationMap):
    with open(semevalFile, 'r') as file:
        text = file.read()
        processedParaList = []
        paragraphs = [s for s in text.split('\n\n') if s]
        for paragraph in paragraphs:
            feature_array = []
            if (paragraph and not paragraph.isspace()):
                paragraph = trim(paragraph)
                paragraphLines = paragraph.split("\n")
                completeSentence = paragraphLines[0]
                relationComponent = paragraphLines[1]
                e1 = re.compile('<e1>(.*?)</e1>').search(completeSentence)
                e_1 = e1.group(1)
                e2 = re.compile('<e2>(.*?)</e2>').search(completeSentence)
                e_2 = e2.group(1)
                completeRelation = ""
                relation_direction = 2
                if (relationComponent != 'Other'):
                    tempSplit = relationComponent.split("(")
                    completeRelation = tempSplit[0]
                    direction = tempSplit[1].replace(")", "")
                    if (direction == E1_E2):
                        relation_direction = E1_to_E2
                    else:
                        relation_direction = E2_to_E1
                else:
                    completeRelation = relationComponent
                preprocessed_sent = completeSentence.replace("<e1>", "") \
                    .replace("</e1>", "").replace("<e2>", "") \
                    .replace("</e2>", "")
                feature_array.append(preprocessed_sent)
                feature_array.append(e_1)
                feature_array.append(e_2)
                feature_array.append(semanticRelationMap.get(completeRelation))
                feature_array.append(relation_direction)
                processedParaList.append(feature_array)
    return processedParaList
