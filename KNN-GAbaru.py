import csv
import math
import operator
import random
import sys
import time
import datetime
from sklearn import preprocessing

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

        count = 1
        for x in range(len(dataset)):
            for y in range(13):
                if dataset[x][y] == "?":
                    dataset[x][y] = float(-9.0)
                else:
                    dataset[x][y] = float(dataset[x][y])

        a, b, c, d, e, f, g, h, i, j, k, l, m, n = zip(*dataset)
        coba = zip(a, b, c, d, e, f, g, h, i, j, k, l, m)
        fixbgt = list(coba)
        fix = [list(elem) for elem in fixbgt]

        normalisasi = preprocessing.normalize(fix)
        a, b, c, d, e, f, g, h, i, j, k, l, m = zip(*normalisasi)

        zip2 = zip(a, b, c, d, e, f, g, h, i, j, k, l, m, n)
        fixbgt1 = list(zip2)
        yep = [list(elem) for elem in fixbgt1]

        for x in range(len(yep)):
            if random.random() < split:
                trainingSet.append((count, yep[x]))
            else:
                testSet.append((count, yep[x]))
            count += 1

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

def getNeighbors(trainingSet, testInstance, k, weights):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length, weights)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(-1))
    neighbors = []
    for x in range(len(distances)):
        if distances[x][1] <= k:
            neighbors.append((distances[x][0], distances[x][1]))
    if neighbors != None:
        neighbors.append((distances[0][0], distances[0][1]))
    return neighbors

def getResponse(neighbors):
    wx = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0][1][-1]
        if neighbors[x][-1] == 0.0:
            return neighbors[x][0][1][-1]
        elif response in wx:
            wx[response] += (1 / pow(neighbors[x][-1], 2))
        else:
            wx[response] = (1 / pow(neighbors[x][-1], 2))
    sortedVotes = sorted(wx.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet) - 1):
        if testSet[x][1][-1] == predictions[x]:
            correct += 1
    return float((correct / float(len(testSet))) * 100.0)

def specy(TrescholdValue, weights):
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
    weights = [0] * 13
    TresholdValue = input("Masukkan jumlah k: ")
    TresholdValue=int(TresholdValue)

    start_time = time.time()
    king = {"Epoch": 0, "Weight": weights, "Akurasi": float(specy(TresholdValue, weights))}
    speed = time.time() - start_time

    Epos = input("Masukkan jumlah epoch: ")
    Epos = int(Epos)
    Deep = 1
    print("Jumlah Epoch = " + str(Epos))
    print("Kedalaman analisis = " + str(Deep))
    execution_time = 1.1 * speed * Epos * Deep * int(5)
    print("Perkiraan waktu eksekusi = " + str(datetime.timedelta(seconds=int(execution_time))))
    print("Perkiraan waktu eksekusi 1 epoch = " + str(datetime.timedelta(seconds=int(execution_time / Epos))))
    print("=======================================================================")
    print("Eksekusi Algoritma Gabungan : ")
    print("=======================================================================")
    print()
    print()
    print("Epoch : 0")
    print("- - - - - - - - - - - - -")
    print("Epoch terbaik :")
    print(king)
    # Write the head of file
    filename = "output.txt"
    with open(filename, 'a') as out:
        str1 = "Jumlah Epoch = " + str(Epos) + "\n" + "Kedalaman analisis = " + str(Deep) + "\n" + \
               "Perkiraan waktu eksekusi = " + str(datetime.timedelta(seconds=int(execution_time))) + "\n " + \
               "Klasifikasi Penyakit Jantung menggunakan Algoritma Genetik dengan KNN " + "\n" + "=" * 20 + "\n"
        out.write(str1)

    # proses GA
    genome = weights
    for epoha in range(Epos):
        organism = []
        for x in range(Deep):
            for k in range(10):
                new_weight = [float(i) for i in[random.SystemRandom().uniform(-1, 1) for _ in range(13)]]
                genome2 = [abs(float((x + y))) for x, y in zip(genome, new_weight)]
                organism.append({"Epoch": epoha, "Weight": genome2, "Akurasi": float(specy(TresholdValue, genome2))})
        prince = max(organism, key=lambda c: c['Akurasi'])
        if prince["Akurasi"] > king["Akurasi"]:
            king = prince
        filename = "output.txt"
        with open(filename, 'a') as out:
            out.write(str(prince) + '\n')
        genome = king["Weight"]
        print(king)
        if king["Akurasi"] > 99.8:
            break
    print()
    print()
    print("=======================================================================")
    print()
    print()
    print("Epoch : " + str(epoha))
    print("- - - - - - - - - - - - -")
    print("Epoch terbaik :")
    print(king)
    with open(filename, 'a') as out:
        str1 = "=" * 20
        str1 = str1 + "\nJumlah Epoch = " + str(Epos) + "\n" + "Kedalaman analisis = " + str(Deep) + "\n" + \
               "Perkiraan waktu eksekusi = " + str(datetime.timedelta(seconds=int(execution_time))) + "\n " + \
               "Klasifikasi Penyakit Jantung menggunakan Algoritma Genetik dengan KNN" + "\n" + "Yang terbaik adalah: \n" + str(
            king) \
               + "\n" + "=" * 20
        out.write(str1)


train()
