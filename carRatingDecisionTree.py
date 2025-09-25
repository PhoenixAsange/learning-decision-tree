import matplotlib.pyplot as plt
from collections import Counter
import random, math, csv

class Node:
    '''
    Node class that the tree is built with
    '''
    def __init__(self, attribute=None, children=None, label=None):
        self.attribute = attribute
        self.children = children or {} # {attributeValue, node}
        self.label = label


class LearningDecisionTree:
    '''
    LearningDecisionTree class that stores a root Node
    '''
    def __init__(self, root=None):
        self.root = root

    def ID3(self, dataset, candidateAttributes, targetIndex):
        '''
        Recursively builds a decision tree
        '''
        
        #Base Case - All labels in a pure subset are identical
        if purityCheck(dataset, targetIndex) == True:
            return Node(label=dataset[0][targetIndex])

        #Calculate the information gain for all possible splits to find best attribute
        bestAttribute = candidateAttributes[0] #Process the first attribute before comparing with others
        bestInformationGain = calculateInformationGain(dataset, bestAttribute, targetIndex)

        #Compare all the possible splits
        for attribute in candidateAttributes[1:]:
            informationGain = calculateInformationGain(dataset, attribute, targetIndex)
            if informationGain > bestInformationGain:
                bestInformationGain = informationGain
                bestAttribute = attribute

        #Create the current node using the best split
        currentNode = Node(attribute=bestAttribute)

        #Pass on the candidate attributes except the one that was split on
        childrenCandidateAttributes = []
        for category in candidateAttributes:
            if category != bestAttribute:
                childrenCandidateAttributes.append(category)

        #Create children for each category
        categoryDatapoints = findCategoryDatapoints(dataset, bestAttribute)
        for attributeValue, datapoints in categoryDatapoints.items():
            childSubtree = self.ID3(datapoints, childrenCandidateAttributes[:], targetIndex) 
            #The datapoints of a category is the new dataset, refining the dataset down 
            # no branches are ever made, just logically as the subset of the dataset follows the split condition 
            # e.g. if safety is med, then the subset that is used in the next recursice iteration is all datapoints where the safety is med
            currentNode.children[attributeValue] = childSubtree #Add to 

        return currentNode
    
class PredictionData:
    '''
    Class that manages the data and metrics of a decision tree's predictions
    '''
    def __init__(self):
        self.trueTargetValueList = []
        self.predictedTargetValueList = []
        self.correctPredictions = 0
        self.accuracy = 0
        self.precisionMacroAverage = 0
        self.recallMacroAverage = 0
        self.f1MacroAverage = 0

    def predictDataset(self, tree, dataset, majorityLabel, targetIndex):
        '''
        Uses a built decision tree to predict the values of a given dataset and data tracking
        '''

        #Predict the target of all datapoints
        for datapoint in dataset:
            #Collect actual value of target
            trueTargetValue = datapoint[targetIndex]
            self.trueTargetValueList.append(trueTargetValue)

            #Make a predicition 
            predictedTargetValue = predict(tree.root, datapoint, majorityLabel)

            #Collect predicted value of target
            self.predictedTargetValueList.append(predictedTargetValue)
            
            #Manage correct predictions
            if predictedTargetValue == trueTargetValue:
                self.correctPredictions += 1

    def calculatePerformanceMetrics(self, category):
        '''
        Calculates precision, recall and f1 of the predictions for a single category
        '''

        #Compute confusion matrix
        truePositives = 0
        falsePositives = 0
        falseNegatives = 0
        for i in range(len(self.trueTargetValueList)):
            if self.trueTargetValueList[i] == category and self.predictedTargetValueList[i] == category:
                truePositives += 1
            elif self.trueTargetValueList[i] != category and self.predictedTargetValueList[i] == category:
                falsePositives += 1
            elif self.trueTargetValueList[i] == category and self.predictedTargetValueList[i] != category:
                falseNegatives += 1

        #Calculate performance metrics
        precision = truePositives / (truePositives + falsePositives) if (truePositives + falsePositives) > 0 else 0
        recall    = truePositives / (truePositives + falseNegatives) if (truePositives + falseNegatives) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def calculateEvaluationMetrics(self, dataset, targetIndex):
        '''
        Calculates the macro-average and weighted average evaluation for a decision tree. 
        Returns a dictonary: {metric, value}.
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
            precision, recall, f1 = self.calculatePerformanceMetrics(category)

            treeLabelCategoriesPrecision[category] = precision
            treeLabelCategoriesRecall[category] = recall
            treeLabelCategoriesF1[category] = f1

            print(f"Class: {category}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-score:  {f1:.3f}\n")

        self.accuracy = self.correctPredictions / len(dataset)

        #Calculate the macro-averages
        self.precisionMacroAverage = calculateMacroAverage(treeLabelCategoriesPrecision)
        self.recallMacroAverage = calculateMacroAverage(treeLabelCategoriesRecall)
        self.f1MacroAverage = calculateMacroAverage(treeLabelCategoriesF1)

        #Calculate the weighted-averages
        self.precisionWeightedAverage = self.calculateWeightedAverage(treeLabelCategoriesPrecision, dataset, targetIndex)
        self.recallWeightedAverage = self.calculateWeightedAverage(treeLabelCategoriesRecall, dataset, targetIndex)
        self.f1WeightedAverage = self.calculateWeightedAverage(treeLabelCategoriesF1, dataset, targetIndex)

        print("Macro averages:")
        print(f"  Precision: {self.precisionMacroAverage:.3f}")
        print(f"  Recall:    {self.recallMacroAverage:.3f}")
        print(f"  F1-score:  {self.f1MacroAverage:.3f}\n")

        print("Weighted averages:")
        print(f"  Precision: {self.precisionWeightedAverage:.3f}")
        print(f"  Recall:    {self.recallWeightedAverage:.3f}")
        print(f"  F1-score:  {self.f1WeightedAverage:.3f}\n")

    def calculateWeightedAverage(self, classMetrics, dataset, targetIndex):
        '''
        Compute the weighted-average of a class metric dictionary
        Σ (support_c * F1_c)) / N
        '''
        supportDictionary = Counter(row[targetIndex] for row in dataset)

        weightedSum = 0.0
        for category in classMetrics:
            support = supportDictionary[category]
            metric = classMetrics[category]
            weightedSum += support * metric

        weightedAverage = weightedSum / len(self.trueTargetValueList)
        return weightedAverage

def calculateMacroAverage(classMetrics):
    '''
    Compute the macro-average of a class metric dictionary
    (Σ F1_c) / |C|
    '''
    total = 0
    for category in classMetrics:
        total += classMetrics[category]
    macroAverage = total / len(classMetrics)
    return macroAverage

def purityCheck(dataset, targetIndex):
    '''
    Returns true if a subset of the data is pure
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

def findCategoryDatapoints(dataset, attributeIndex):
    '''
    Finds the datapoints of all categories of an attribute
    '''
    attributeDatapointsDictonary = {} #{category, [datapoints]}
    for datapoint in dataset:
        attribute = datapoint[attributeIndex]
        if attribute not in attributeDatapointsDictonary:
            attributeDatapointsDictonary[attribute] = []
        attributeDatapointsDictonary[attribute].append(datapoint)
    return attributeDatapointsDictonary

def calculateInformationGain(dataset, attributeIndex, targetIndex):
    '''
    Calculates the information gain of an attribute selection    
    '''
    datasetEntropy = calculateEntropy(dataset, targetIndex) #H(S)

    # Calculate entropy after an attribute selection
    partition = findCategoryDatapoints(dataset, attributeIndex)
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
    treePredictionData.predictDataset(decisionTree, testingData, treeTrainingDataMajorityLabel, targetIndex)
    treePredictionData.calculateEvaluationMetrics(testingData, targetIndex)

    # Print sizes
    print("Total dataset size:", len(dataset))
    print("Training dataset size:", len(trainingData))
    print("Testing dataset size:", len(testingData))
    print(f"\nLearning tree accuracy: {treePredictionData.accuracy:.3f}\n")
    iterationData = {"accuracy": treePredictionData.accuracy,
    "macro-precision": treePredictionData.precisionMacroAverage,
    "macro-recall":    treePredictionData.recallMacroAverage,
    "macro-f1":        treePredictionData.f1MacroAverage,
    "weighted-precision": treePredictionData.precisionWeightedAverage,
    "weighted-recall":    treePredictionData.recallWeightedAverage,
    "weighted-f1":        treePredictionData.f1WeightedAverage}

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

    #Create multiple trees (trees = increments) using differently sized subsets of the training data
    targetIndex = -1
    increments = 40
    treeData = {}

    for i in range(1, increments + 1) :
        datasetPercent = i / increments
        trainingDatapointAmount = max(1, int(datasetPercent * len(carTrainingData)))  

        #Limit training data
        trainingDataSubset = carTrainingData[:trainingDatapointAmount]  

        print(f"\nTree with {datasetPercent:.0%} training data")

        #Make predictions about testing dataset using the subset of the traingdata
        iterationData = buildAndPredictTree(carData, trainingDataSubset, carTestingData, targetIndex)
        
        #Add iteration data for graph later
        treeData[datasetPercent * 100] = iterationData

    #Create metric graph
    plotDecisionTreeData(treeData)
    
def plotDecisionTreeData(treeData):
    # Sort by training percentage
    percents = sorted(treeData.keys())

    # Extract each metric into its own list
    accuracy           = [treeData[p]["accuracy"] for p in percents]
    macro_precision    = [treeData[p]["macro-precision"] for p in percents]
    macro_recall       = [treeData[p]["macro-recall"] for p in percents]
    macro_f1           = [treeData[p]["macro-f1"] for p in percents]
    weighted_precision = [treeData[p]["weighted-precision"] for p in percents]
    weighted_recall    = [treeData[p]["weighted-recall"] for p in percents]
    weighted_f1        = [treeData[p]["weighted-f1"] for p in percents]

    plt.figure(figsize=(8, 5))
    plt.plot(percents, accuracy,           marker="o", label="Accuracy")
    plt.plot(percents, macro_precision,    marker="o", label="Macro Precision")
    plt.plot(percents, macro_recall,       marker="o", label="Macro Recall")
    plt.plot(percents, macro_f1,           marker="o", label="Macro F1")
    plt.plot(percents, weighted_precision, marker="o", label="Weighted Precision")
    plt.plot(percents, weighted_recall,    marker="o", label="Weighted Recall")
    plt.plot(percents, weighted_f1,        marker="o", label="Weighted F1")

    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Score")
    plt.title(f"Decision Tree Metrics vs Training Data ({len(treeData)} iterations)")
    plt.grid(True)
    plt.legend()
    plt.savefig("car-tree-metrics1.png")

if __name__ == "__main__":
    main()