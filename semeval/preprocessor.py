from semeval.utils import *

def getMaxSentenceLengthInTraining():
    printConsole("In Function - getMaxSentenceLengthInTraining()")
    with open(TRAINING_FILE_NAME, 'r') as file:
        text = file.read()
        paragraphs = [s for s in text.split('\n\n') if s]
        maxLength = 0
        for paragraph in paragraphs:
            if (paragraph and not paragraph.isspace()):
                paragraph = trim(paragraph)
                paragraphLines = paragraph.split("\n")
                completeSentence = paragraphLines[0]
                completeSentence = completeSentence.replace("\"",'')
                completeSentence = completeSentence.replace("<e1>",'')
                completeSentence = completeSentence.replace("</e1>", '')
                completeSentence = completeSentence.replace("<e2>", '')
                completeSentence = completeSentence.replace("</e2>", '')
                while(completeSentence[0].isdigit()):
                    completeSentence = completeSentence.replace(completeSentence[0],'')
                sentence = trim(completeSentence) #Plain Sentence Obtained
                currentSentenceLength = len(sentence.split())
                if(currentSentenceLength > maxLength):
                    maxLength = currentSentenceLength
        printConsole("Max Length of Sentence Is:"+str(maxLength))
        return maxLength


def getPreProcessedRelationMap():
    printConsole("In Function - getPreProcessedRelationMap()")
    relationshipSet = set()
    with open(TRAINING_FILE_NAME, 'r') as file:
        text = file.read()
        paragraphs = [s for s in text.split('\n\n') if s]
        for paragraph in paragraphs:
            if (paragraph and not paragraph.isspace()):
                paragraph = trim(paragraph)
                paragraphLines = paragraph.split("\n")
                relationshipAndDirection = paragraphLines[1]
                relationship = relationshipAndDirection.split("(")[0]
                relationship = trim(relationship)
                relationshipSet.add(relationship)
        #print(relationshipSet)
        printConsole("Extracted All Relationships In Set")
        relationshipToIndexMap = {}
        indexToRelationshipMap = {}
        relationMapValueCounter = 1
        for relation in relationshipSet:
            relationshipToIndexMap[relation]= relationMapValueCounter
            indexToRelationshipMap[relationMapValueCounter] =relation
            relationMapValueCounter = relationMapValueCounter + 1
        #print(relationshipToIndexMap)
        #print(indexToRelationshipMap)
        printConsole("Transformed Relationship Set to relationshipToIndexMap is: ")
        printConsole(relationshipToIndexMap)
        printConsole("Transformed Relationship Set to indexToRelationshipMap is: ")
        printConsole(indexToRelationshipMap)
        printConsole("Returning relationshipToIndexMap ")
        return relationshipToIndexMap