# copyright by safiravanillia

import csv
import random
import operator

#record=[]

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        count = 1
        for x in range(len(dataset)):
            for y in range(14):
                if dataset[x][y ]== "?":
                    dataset[x][y] = float(-9.0)
                else:
                    dataset[x][y] = float(dataset[x][y])
            if random.random() > split:
                trainingSet.append((count, dataset[x]))
                #record.append(x)
                #print(random.random())
            else:
                testSet.append((count, dataset[x]))
            count += 1

def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance

def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = manhattanDistance(testInstance, trainingSet[x][1], length)
        distances.append(((trainingSet[x][0], trainingSet[x][1], dist)))
    distances.sort(key=operator.itemgetter(-1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][1][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    trainingSet = []
    testSet = []
    split = 0.3
    loadDataset('data.csv', split, trainingSet, testSet)
    print("Train set:" + repr(len(trainingSet)))
    print("Test set:"+ repr(len(testSet)))

    predictions = []
    correct = 0
    false = 0
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x][1], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> Record of Test Set=' + repr(testSet[x][0]) + ', predicted=' + repr(result) + ', actual=' + repr(testSet[x][1][-1]))
        for y in range(k):
            print('Neighbours (rec: ' + repr(neighbors[y][0]) +') =' + repr(neighbors[y][1]) + ', Distance=' + repr(neighbors[y][-1]))
        print("")
        if testSet[x][1][-1] == result:
            correct+=1
        else:
            false+=1

    print('\nCorrect prediction: ' + repr(correct))
    print('False prediction: ' + repr(false))

    accuracy = getAccuracy(testSet, predictions)
    print('\nAccuracy: ' + repr(accuracy) + '%')

main()
