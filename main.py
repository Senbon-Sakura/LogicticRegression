import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadDataSet(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append(lineArr[:-1])
        labelArr.append(lineArr[-1])
    return np.array(dataArr, dtype='float32'), np.array(labelArr, dtype='float32').reshape(-1,1)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def GradDescent(dataArr, labelArr, lr=0.2, maxIter=500):
    sampNum, featNum = dataArr.shape
    W = np.random.randn(featNum, 1) * 0.1
    b = 0.0
    GradWList = []
    GradbList = []
    for iter in range(maxIter):
        Z = np.dot(dataArr, W) + b
        fx = sigmoid(Z)
        a = dataArr * (labelArr - fx)
        GradW = -1*sum( dataArr * (labelArr - fx) ) / sampNum
        Gradb = -1*sum( labelArr - fx ) / sampNum
        W -= lr * GradW.reshape(-1,1)
        b -= lr * Gradb
        GradWList.append(GradW)
        GradbList.append(Gradb)
    return W, b, GradWList, GradbList

dataArr, labelArr = loadDataSet('testSet.txt')
W, b, GradWList, GradbList = GradDescent(dataArr, labelArr)
print(W, b)
fig = plt.figure()
#ax3 = plt.axes(projection='3d')
fx = sigmoid( np.dot(dataArr, W) + b )
#ax3.scatter3D(dataArr[:,0], dataArr[:,1], s=(labelArr+2)+30, c=100*(labelArr+2)+100)
plt.scatter(dataArr[:,0], dataArr[:,1], s=(labelArr.reshape(-1)*80)+40, c=100*(labelArr.reshape(-1)+2)+100)
x = np.linspace(min(dataArr[:,0]), max(dataArr[:,0]), 100)
plt.plot(x, -1*W[0]/W[1]*x+b)

fig2 = plt.figure()
xAxis = list(range(len(GradbList)))
plt.subplot(131)
plt.plot(xAxis, np.array(GradWList)[:,0])
plt.subplot(132)
plt.plot(xAxis, np.array(GradWList)[:,1])
plt.subplot(133)
plt.plot(xAxis, np.array(GradbList)[:,0])
#plt.plot(xAxis, GradbList)


plt.show()




