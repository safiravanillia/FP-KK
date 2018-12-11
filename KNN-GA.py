import csv
import math
import operator
import random
import sys
import time
import datetime

# Handling dataset dari iris.data dan split menjadi trainingSet dan testSet
def loadDataset(filename, split, trainingSet=[], testSet=[]):
 with open(filename, 'rb') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)-1):
        for y in range(13):
            if dataset[x][y] == "?":
                dataset[x][y] = float(-9.0)
            else:
                dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])


# perhitungan euclidean distance untuk menghitung jarak -- terikat dengan getNeighbors
def euclideanDistance(instance1, instance2, length, weights):
    distance = 0
    for x in range(length):
        distance += (weights[x] * pow((instance1[x] - instance2[x]), 2))
    try:
        return math.sqrt(distance)
    except ValueError:
        print(distance)
        print(weights)
        sys.exit(0)

# tujuan : cari lokasi 'k most similar' data instances di training set, pada test set untuk diprediksi nantinya
def getNeighbors(trainingSet, testInstance, k, weights):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length, weights)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(len(distances)):
        if distances[x][1] <= k:
            neighbors.append(distances[x][0])
    if neighbors != None:
        neighbors.append(distances[0][0])
    return neighbors

# menyusun perkiraan respon berdasarkan neighbors tersebut
# neighbors ambil vote untuk atribut class-nya dan getResponse akan ambil vote terbesar sebagai prediksi
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# evaluasi accuracy dari prediksi - dibandingkan dengan testSet awalnya
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet) - 1):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return float((correct / float(len(testSet))) * 100.0)

# algoritma klasifikasi k-nn
def specy(TrescholdValue, weights):
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('data.csv', split, trainingSet, testSet)
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], TrescholdValue, weights)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    return accuracy


def train():
    # weight = 4 karena jumlah atribut pada iris.data adalah 4
    weights = [0] * 13
    # TresholdValue -- threshold value k* is used to segment an image, digunakan 0.2 dari range 0 - 1
    TresholdValue = input("Masukkan jumlah k: ")
    start_time = time.time()
    king = {"Epoch": 0, "Genome": weights, "Accuracy": float(specy(TresholdValue, weights))}
    speed = time.time() - start_time
    # epos sejumlah 50x
    Epos = input("Masukkan jumlah epoch: ")
    Deep = 1
    print("Number of epochs is " + str(Epos))
    print("Deepness of the analysis is " + str(Deep))
    execution_time = 1.1 * speed * Epos * Deep * int(5)
    print("Estimated execution time is " + str(datetime.timedelta(seconds=int(execution_time))))
    print("Estimated execution time for one epoch is " + str(
        datetime.timedelta(seconds=int(execution_time / 50))))
    print("=======================================================================")
    print("Starting evolutional algorithm: ")
    bb = 0
    print("=======================================================================")
    print()
    print()
    print("Epoch num is 0")
    print("- - - - - - - - - - - - -")
    print("The best is:")
    print(king)
    # Write the head of file
    filename = "output.txt"
    with open(filename, 'a') as out:
        str1 = "Number of epochs is " + str(Epos) + "\n" + "Deepness of the analysis is " + str(Deep) + "\n" + \
               "Estimated execution time is " + str(datetime.timedelta(seconds=int(execution_time))) + "\n " + \
               "Genetic algorithms optimisation for KNN with preference weights" + "\n" + "=" * 20 + "\n"
        out.write(str1)

    # Execute GA
    genome = weights
    for epoha in range(Epos):
        organism = []
        for x in range(Deep):
            for k in range(10):
                # range = 4 -- uniform crossover
                new_weight = [float(i) for i in
                              [random.SystemRandom().uniform(-10.0, 10.0) for _ in range(13)]]                               #generate random data for population
                #print (new_weight)
                genome2 = [abs(float((x + y))) for x, y in zip(genome, new_weight)]
                #print (x,y)
                organism.append({"Epoch": epoha, "Genome": genome2, "Accuracy": float(specy(TresholdValue, genome2))})      #getting accuracy with knn classification
        prince = max(organism, key=lambda c: c['Accuracy'])                                                                 #prince as temporary better accuracy
        if prince["Accuracy"] > king["Accuracy"]:                                                                           #king as the best accuracy
            king = prince                                                                                                    #the king is not greater than 99.8
        filename = "output.txt"                                                                                             #the best king is what we looking for
        with open(filename, 'a') as out:
            out.write(str(prince) + '\n')
        genome = king["Genome"]
        #bb = 100 * epoha / Epos
        print(king)
        if king["Accuracy"] > 99.8:
            break
    print()
    print()
    print("=======================================================================")
    print()
    print()
    print("Epoch num is " + str(epoha))
    print("- - - - - - - - - - - - -")
    print("The best is:")
    print(king)
    print("That is all!")
    with open(filename, 'a') as out:
        str1 = "=" * 20
        str1 = str1 + "\nNumber of epochs is " + str(Epos) + "\n" + "Deepness of the analysis is " + str(Deep) + "\n" + \
               "Estimated execution time is " + str(datetime.timedelta(seconds=int(execution_time))) + "\n " + \
               "Genetic algorithms optimisation for KNN with preference weights" + "\n" + "The king is: \n" + str(king) \
               + "\n" + "=" * 20
        out.write(str1)


train()

