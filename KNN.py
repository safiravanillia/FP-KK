import csv
import random
import operator
import math
from sklearn import preprocessing

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

        count = 1
        for x in range(len(dataset)):
            for y in range(13):
                if dataset[x][y ]== "?":
                    dataset[x][y] = float(-9.0)
                else:
                    dataset[x][y] = float(dataset[x][y])

        a, b, c, d, e, f, g, h, i, j, k, l,m, n = zip(*dataset)
        coba = zip(a, b, c, d, e, f, g, h, i, j, k, l,m )
        fixbgt = list(coba)
        fix = [list(elem) for elem in fixbgt]

        normalisasi = preprocessing.normalize(fix)
        a, b, c, d, e, f, g, h, i, j, k, l, m = zip(*normalisasi)

        zip2=zip(a, b, c, d, e, f, g, h, i, j, k, l,m,n)
        fixbgt1 = list(zip2)
        yep = [list(elem) for elem in fixbgt1]

        for x in range(len(yep)):
            if random.random() < split:
                trainingSet.append((count, yep[x]))
            else:
                testSet.append((count, yep[x]))
            count += 1

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(abs(instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x][1], length)
        distances.append(((trainingSet[x][0], trainingSet[x][1], dist)))
    distances.sort(key=operator.itemgetter(-1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors

def getResponse(neighbors):
    wx = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1][-1]
        if neighbors[x][-1] == 0.0:
            return neighbors[x][1][-1]
        elif response in wx:
            wx[response] += (1/pow(neighbors[x][-1], 2))
        else:
            wx[response] = (1/pow(neighbors[x][-1], 2))
    sortedVotes = sorted(wx.items(), key=operator.itemgetter(1), reverse=True)
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
    split = 0.67
    loadDataset('data.csv', split, trainingSet, testSet)
    print("Data uji:" + repr(len(trainingSet)))
    print("Data tes:"+ repr(len(testSet)))

    predictions = []
    correct = 0
    false = 0
    k=input("\nMasukkan nilai k: ")
    k=int(k)
    for x in range(len(testSet)):
        print("=========================================================================")
        print("Jumlah k sebanyak " + str(k))
        neighbors = getNeighbors(trainingSet, testSet[x][1], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> Urutan data uji=' + repr(testSet[x][0]))
        print('> Prediksi=' + repr(result))
        print('> Sebenarnya=' + repr(testSet[x][1][-1]))
        for y in range(k):
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print('k ke-' + str(y))
            print('\nTetangga (Urutan: ' + repr(neighbors[y][0]) +') \n=' + repr(neighbors[y][1]))
            print('\nJarak=' + repr(neighbors[y][-1]))
        if testSet[x][1][-1] == result:
            correct+=1
        else:
            false+=1

    print("=============================================================================")
    print('\nJumlah prediksi benar: ' + repr(correct))
    print('Jumlah prediksi salah: ' + repr(false))

    accuracy = getAccuracy(testSet, predictions)
    print('\nAkurasi: ' + repr(accuracy) + '%')

main()
