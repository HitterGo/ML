from numpy import *

from speakingDetection import SVM

################## test svm #####################  
## step 1: load data  
print('step 1: load data...')
dataSet = []  
labels = []  
fileIn = open('training.data')
for line in fileIn.readlines():  
    lineArr = line.strip().split(' ')
    dataSet.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]), float(lineArr[3]),
    				float(lineArr[4])])
    labels.append(float(lineArr[5]))  
  
dataSet = mat(dataSet)  
labels = mat(labels).T  
train_x = dataSet[1:1000, :]
train_y = labels[1:1000, :]
test_x = dataSet[1001:2000, :]
test_y = labels[1001:2000, :]
  
## step 2: training...  
print('step 2: training...')
C = 0.6  
toler = 0.001
maxIter = 50  
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

## step 3: testing  
print('step 3: testing...')
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)  
print(accuracy)
'''
## step 4: show the result  
print "step 4: show the result..."    
SVM.showSVM(svmClassifier)  
'''