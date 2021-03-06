import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        print(h.shape)
        error = (labelMat-h)
        #print("weights shape:\t" + str(weights.shape))
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    weightsList = []
    for i in range(200):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i]-h
            weights = weights + alpha*error*dataMatrix[i]
            weightsList.append(weights)
    return weights, weightsList

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    weightsList = []
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha*error*dataMatrix[randIndex]
            weightsList.append(weights)
            del (dataIndex[randIndex])
    return weights, weightsList

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    y = np.array(y).flatten()
    ax.plot(x,y.flatten())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append((lineArr))
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))





#'''
dataArr, labelArr = loadDataSet()
#weights = gradAscent(dataArr, labelArr)
weightsList = []
weights0, weightsList0 = stocGradAscent0(np.array(dataArr), labelArr)
weights1, weightsList1 = stocGradAscent1(np.array(dataArr), labelArr)
plotBestFit(weights1)
fig = plt.figure(num=2)
plt.subplot(231)
plt.plot(range(np.shape(weightsList0)[0]),[X[0] for X in weightsList0])
plt.subplot(232)
plt.plot(range(np.shape(weightsList0)[0]),[X[1] for X in weightsList0])
plt.subplot(233)
plt.plot(range(np.shape(weightsList0)[0]),[X[2] for X in weightsList0])
plt.subplot(234)
plt.plot(range(np.shape(weightsList1)[0]),[X[0] for X in weightsList1])
plt.subplot(235)
plt.plot(range(np.shape(weightsList1)[0]),[X[1] for X in weightsList1])
plt.subplot(236)
plt.plot(range(np.shape(weightsList1)[0]),[X[2] for X in weightsList1])
plt.show()
#'''
#multiTest()
