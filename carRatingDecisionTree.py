import matplotlib.pyplot as plt
from collections import Counter
import random, math, csv

class Node:
    def __init__(self, attribute=None, children=None, label=None):
        self.attribute = attribute
        self.children = children or {} # {featureValue, node}
        self.label = label


class LearningDecisionTree:
    def __init__(self, root=None):
        self.root = root

    def ID3(self, dataset, candidateAttributes, targetIndex):
        '''
        Build and return a subtree using ID3
        '''
        #Base cases
        if len(dataset) == 0:
            return Node()
        
        if purityCheck(dataset, targetIndex) == True:
            return Node(label=dataset[0][targetIndex])
        
        #Base case for no attributes left
        #Base case for no good split
        
        #Calculate the information gain for all possible splits to calculate best attribute
        bestAttribute = candidateAttributes[0]
        bestInformationGain = calculateInformationGain(dataset, bestAttribute, targetIndex)
        for attribute in candidateAttributes[1:]:
            informationGain = calculateInformationGain(dataset, attribute, targetIndex)
            if informationGain > bestInformationGain:
                bestInformationGain = informationGain
                bestAttribute = attribute

        if bestInformationGain <= 0:
            return #No useful split, return majority label 

        # Create the children nodes
        currentNode = Node(attribute=bestAttribute)
        partitions = partitionAttribute(dataset, bestAttribute)
        childrenCandidateAttributes = []
        for attribute in candidateAttributes:
            if attribute != bestAttribute:
                childrenCandidateAttributes.append(attribute)

        for attributeValue, subset in partitions.items():
            childSubtree = self.ID3(subset, childrenCandidateAttributes[:], targetIndex)
            currentNode.children[attributeValue] = childSubtree

        return currentNode

def purityCheck(dataset, targetIndex):
    '''
    Returns true if a subset of the data is pure (only includes one category of the Y value)
    '''
    return calculateEntropy(dataset, targetIndex) == 0

def calculateEntropy(dataset, columnIndex):
    '''
    Calculate the entropy of a datasets column 
    '''

    #Find how many categories there are and how many occurances each has
    countDictionary = Counter(row[columnIndex] for row in dataset) #{category, samples (amount of rows for the given category)}
    datasetSampleAmount = len(dataset) #The amount of samples (rows) in the dataset

    #Calculate entropy using the H(S) formula
    sumProbabilityLog = 0.0
    for samples in countDictionary.values(): # Σ
        probability = samples / datasetSampleAmount # p(x)
        sumProbabilityLog += probability * math.log2(probability) #p(x) * log2(p(x))
    entropy = -sumProbabilityLog # Negation

    return entropy

def partitionAttribute(dataset, attributeIndex):
    '''
    Split the dataset on a given attribute
    '''
    partitions = {} #{partitionKey, [keyValue]}
    for row in dataset:
        partitionKey = row[attributeIndex]
        if partitionKey not in partitions:
            partitions[partitionKey] = []
        partitions[partitionKey].append(row)
    return partitions

def calculateInformationGain(dataset, attributeIndex, targetIndex):
    '''
    Calculates the information gain of an attribute selection    
    '''
    datasetEntropy = calculateEntropy(dataset, targetIndex) #H(S)

    # Calculate entropy after an attribute selection
    partition = partitionAttribute(dataset, attributeIndex)
    totalDatasetRows = len(dataset)

    # Calculate IG(A,S)
    weightedEntropySum = 0.0
    for subsetRows in partition.values(): # Σ_vϵvalues(A)
        weightedEntropySum += len(subsetRows) / totalDatasetRows * calculateEntropy(subsetRows, targetIndex) # (|S_v|/|S|* H(S_v))

    informationGain = datasetEntropy - weightedEntropySum
    return informationGain

def majorityLabel(dataset, targetIndex):
    '''
    Finds the label with the 
    '''
    labelCounts = Counter()
    for row in dataset:
        label = row[targetIndex]
        labelCounts[label] += 1

    # Get the most common label using Counter.most_common
    mostCommonList = labelCounts.most_common(1)  # returns [(label, count)]
    majorityLabel = mostCommonList[0][0]

    return majorityLabel

def main():
    # Prepare the data
    with open("car.csv", "r") as carDataFile:
        carDataReader = csv.reader(carDataFile) 
        carData = list(carDataReader) # Extract the data into a list of data points

    carData = carData[1:] # Remove the header
    random.shuffle(carData)

    splitIndex = len(carData) // 2
    trainingData = carData[:splitIndex]
    testingData = carData[splitIndex:]

    targetIndex = -1

    #Calculate candidate attributes
    candidateAttributes = []
    for attribute in range(len(trainingData[0]) - 1):
        candidateAttributes.append(attribute)

    #Build the tree using ID3
    carRatingDecisionTree = LearningDecisionTree()
    carRatingDecisionTree.root = carRatingDecisionTree.ID3(trainingData, candidateAttributes, targetIndex)

if __name__ == "__main__":
    main()