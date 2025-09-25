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
    
class PredictionData:
    def __init__(self):
        self.trueTargetValueList = []
        self.predictedTargetValueList = []
        self.correctPredictions = 0
        self.accuracy = 0
        self.macroAverage = 0
        self.weightedAverage = 0

    def predictDataset(self, tree, dataset, majorityLabel):
        for datapoint in dataset:
            #Collect accuracy data while predicting
            trueTargetValue = datapoint[-1]
            self.trueTargetValueList.append(trueTargetValue)

            predictedTargetValue = predict(tree.root, datapoint, majorityLabel)
            self.predictedTargetValueList.append(predictedTargetValue)
            
            if predictedTargetValue == trueTargetValue:
                self.correctPredictions += 1

    def calculateF1(self, category, trueTargetValueList, predictedTargetValueList):
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

    def calculateMetrics(self, dataset, targetIndex):
        '''
        Calculates metrics for a decision tree. Returns a dictonary: {metric, value}
        '''
        #Find the labels categories
        categories = []
        for row in dataset:  
            labelCategory = row[targetIndex]
            if labelCategory not in categories:
                categories.append(labelCategory)

        #Calculate precision, recall and f1 for each category
        treeLabelCategoriesPrecision = {} #{category, precision}
        treeLabelCategoriesRecall = {} #{category, recall}
        treeLabelCategoriesF1 = {} #{category, f1}

        for category in categories:
            precision, recall, f1 = self.calculateF1(category, self.trueTargetValueList, self.predictedTargetValueList)

            treeLabelCategoriesPrecision[category] = precision
            treeLabelCategoriesRecall[category] = recall
            treeLabelCategoriesF1[category] = f1

            print(f"Class: {category}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-score:  {f1:.3f}\n")

        self.accuracy = self.correctPredictions / len(dataset)

        #Calculate the macro-average
        f1Sum = 0
        for category in treeLabelCategoriesF1:
            f1Sum += treeLabelCategoriesF1[category]
        self.macroAverage = f1Sum / len(treeLabelCategoriesF1)  

        #Calculate the weighted-average
        weightedF1Sum = 0
        supportDictonary = Counter(row[targetIndex] for row in dataset)
        for category in treeLabelCategoriesF1:
            support = supportDictonary[category]
            f1 = treeLabelCategoriesF1[category]
            weightedF1Sum += support * f1
        self.weightedAverage = weightedF1Sum / len(self.trueTargetValueList)

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

def buildAndPredictTree(dataset, trainingData, testingData, targetIndex):
    #Calculate candidate attributes
    candidateAttributes = []
    for attribute in range(len(trainingData[0]) - 1):
        candidateAttributes.append(attribute)

   # Build the tree using ID3
    decisionTree = LearningDecisionTree()
    decisionTree.root = decisionTree.ID3(trainingData, candidateAttributes, targetIndex)

    # Majority label from training set
    treeTrainingDataMajorityLabel = calculateMajorityLabel(trainingData, targetIndex)

    # Run predictions on the test set and collect data
    treePredictionData = PredictionData()
    treePredictionData.predictDataset(decisionTree, testingData, treeTrainingDataMajorityLabel)
    treePredictionData.calculateMetrics(testingData, targetIndex)

    # Print sizes
    print("Total dataset size:", len(dataset))
    print("Training dataset size:", len(trainingData))
    print("Testing dataset size:", len(testingData))
    print(f"\nLearning tree accuracy: {treePredictionData.accuracy:.3f}\n")
    print(f"Macro Average: {treePredictionData.macroAverage}")
    print(f"Weighted Average: {treePredictionData.weightedAverage}")
    iterationData = {"accuracy": treePredictionData.accuracy,
                        "macro-average": treePredictionData.macroAverage,
                        "weighted-average": treePredictionData.weightedAverage}

    return iterationData

def main():
    # Prepare the data
    with open("car.csv", "r") as carDataFile:
        carDataReader = csv.reader(carDataFile) 
        carData = list(carDataReader) # Extract the data into a list of data points

    carData = carData[1:] # Remove the header
    random.shuffle(carData)

    splitIndex = int(0.7 *len(carData)) #Split 70%
    carTrainingData = carData[:splitIndex]
    carTestingData = carData[splitIndex:]

    targetIndex = -1
    increments = 40

    #Should build a tree with a percent of all the training data, print the result. 
    #It does this until all the training data has been used
    treeData = {}
    for i in range(1, increments + 1) :
        datasetSubsetPercent = i / increments
        trainingDatapointAmount = max(1, int(datasetSubsetPercent * len(carTrainingData)))  

        # Slice or sample the training data
        trainingData = carTrainingData[:trainingDatapointAmount]  

        print(f"\nTree with {datasetSubsetPercent:.0%} training data")
        iterationData = buildAndPredictTree(carData, trainingData, carTestingData, targetIndex)
        treeData[datasetSubsetPercent * 100] = iterationData

    #Create metric graph
    plotDecisionTreeData(treeData)
    
def plotDecisionTreeData(treeData):
    # Sort by training percentage
    percents = sorted(treeData.keys())

    # Extract each metric into its own list
    accuracy = [treeData[p]["accuracy"] for p in percents]
    macro    = [treeData[p]["macro-average"] for p in percents]
    weighted = [treeData[p]["weighted-average"] for p in percents]

    # Plot all three lines
    plt.figure(figsize=(8, 5))
    plt.plot(percents, accuracy, marker="o", label="Accuracy")
    plt.plot(percents, macro, marker="o", label="Macro F1")
    plt.plot(percents, weighted, marker="o", label="Weighted F1")

    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Score")
    plt.title(f"Decision Tree Metrics vs Training Data ({len(treeData)} iterations)")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("car-tree-metrics1.png")

if __name__ == "__main__":
    main()