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
        
        #Base Case - All labels in a pure subset are identical
        if purityCheck(dataset, targetIndex) == True:
            return Node(label=dataset[0][targetIndex])

        #Calculate the information gain for all possible splits to calculate best attribute
        bestAttribute = candidateAttributes[0]
        bestInformationGain = calculateInformationGain(dataset, bestAttribute, targetIndex)
        for attribute in candidateAttributes[1:]:
            informationGain = calculateInformationGain(dataset, attribute, targetIndex)
            if informationGain > bestInformationGain:
                bestInformationGain = informationGain
                bestAttribute = attribute

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

def calculateMajorityLabel(dataset, targetIndex):
    """
    Returns the majority class label from the dataset.
    """
    labels = []
    for row in dataset:
        labels.append(row[targetIndex])
    
    # Count how many times each label appears
    labelCounts = Counter(labels)  

    majorityLabel = labelCounts.most_common(1)[0][0]
    return majorityLabel

def predict(node, datapoint, majorityLabel):
    '''
    Recursive prediction algorithm that traverses the tree 
    and returns the predicted label for the datapoint
    '''
    
    #Handles if a branch has an attribute that was not in the training data
    if node is None:
        return majorityLabel

    #Base case
    if node.label is not None:
        return node.label
    
    attributeValue = datapoint[node.attribute]

    #Recurse until a leaf is reached
    return predict(node.children.get(attributeValue), datapoint, majorityLabel)

def calculateF1(category, trueTargetValueList, predictedTargetValueList):
    '''
    Calculates the F1 of the predictions for a single category
    '''
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0

    #Compare the lists to generate the TP, FP and FN
    if len(trueTargetValueList) != len(predictedTargetValueList):
        print("UNEVEN")

    for i in range(len(trueTargetValueList)):
        if trueTargetValueList[i] == category and predictedTargetValueList[i] == category:
            truePositives += 1
        elif trueTargetValueList[i] != category and predictedTargetValueList[i] == category:
            falsePositives += 1
        elif trueTargetValueList[i] == category and predictedTargetValueList[i] != category:
            falseNegatives += 1

    # Precision
    if (truePositives + falsePositives) > 0:
        precision = truePositives / (truePositives + falsePositives)
    else:
        precision = 0

    # Recall
    if (truePositives + falseNegatives) > 0:
        recall = truePositives / (truePositives + falseNegatives)
    else:
        recall = 0

    # F1
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

def main():
    # Prepare the data
    with open("car.csv", "r") as carDataFile:
        carDataReader = csv.reader(carDataFile) 
        carData = list(carDataReader) # Extract the data into a list of data points

    dataAttributes = carData[0]  #Store the header row as the list of attributes
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

    #Make predicitions using the created tree
    carTrainingDataMajorityLabel = calculateMajorityLabel(trainingData, targetIndex)

    trueTargetValueList = []
    predictedTargetValueList = []
    correctPredictions = 0
    for datapoint in testingData:
        #Collect accuracy data while predicting
        trueTargetValue = datapoint[targetIndex]
        trueTargetValueList.append(trueTargetValue)

        predictedTargetValue = predict(carRatingDecisionTree.root, datapoint, carTrainingDataMajorityLabel)
        predictedTargetValueList.append(predictedTargetValue)
        
        if predictedTargetValue == trueTargetValue:
            correctPredictions += 1

    carRatingDecisionTreeAccuracy = correctPredictions / len(testingData)

    #Print the testing data
    #The size of the training and testing sets
    print("Total dataset size:", len(carData))
    print("Training dataset size:", len(trainingData))
    print("Testing dataset size:", len(testingData))

    #Total Accuracy
    print(f"\nLearning tree accuracy: {carRatingDecisionTreeAccuracy}\n")

    #Precision, recall and F1-score values for each class
    #Find the labels categories
    carLabelCategories = []
    for row in carData[1:]:  
        labelCategory = row[targetIndex]
        if labelCategory not in carLabelCategories:
            carLabelCategories.append(labelCategory)

    #Calculate precision, recall and f1 for each category
    carLabelCategoriesPrecision = {} #{category, precision}
    carLabelCategoriesRecall = {} #{category, recall}
    carLabelCategoriesF1 = {} #{category, f1}

    for category in carLabelCategories:
        precision, recall, f1 = calculateF1(category, trueTargetValueList, predictedTargetValueList)

        carLabelCategoriesPrecision[category] = precision
        carLabelCategoriesRecall[category] = recall
        carLabelCategoriesF1[category] = f1

        print(f"Class: {category}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-score:  {f1:.3f}\n")

    #Calculate the macro-average

    #Plot the learning curve
        #Implement accuracy while learning

if __name__ == "__main__":
    main()