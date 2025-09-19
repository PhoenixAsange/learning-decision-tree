import matplotlib.pyplot as plt
from collections import Counter
import random
import csv

class Node:
    def __init__(self, attribute=None, children=None, label=None):
        self.attribute = attribute
        self.children = children or {} # {featureValue, node}
        self.label = label


class LearningDecisionTree:
    def __init__(self, root=None):
        self.root = root

# Extract and randomly split the data
with open("car.csv", "r") as carDataFile:
    reader = csv.reader(carDataFile) 
    carData = carDataFile.readlines() # Extract the data into a list of data points

random.shuffle(carData)

splitIndex = len(carData) // 2
trainingData = carData[:splitIndex]
testingData = carData[splitIndex:]

#Train (ID3)

#Calculate the entropy of a datasets column 
def calculateEntropy(dataset, columnIndex):

    #Find how many categories there are and how many occurances each has
    countDictionary = Counter(row[columnIndex + 1] for row in dataset) #{category, samples}
    datasetSamples = len(dataset) #The amount of samples (rows) in the dataset

    #Calculate entropy H(S)

    return entropy

#Training the tree