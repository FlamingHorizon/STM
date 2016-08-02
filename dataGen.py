# prepare data for linear STM training
# yyq, 2016-01-14

import cPickle
import numpy as np

totalNum = 50000
totalNumTest = 50000
maxLen = 5
minLen = 5
dim = 10
saveFileTrain = 'trainData5w.pkl'
saveFileTest = 'testData5w.pkl'

rng = np.random
trainData = []
trainQuery = []
trainTarget = []
testData = []
testQuery = []
testTarget = []

for i in range(totalNum):
	seqLen = rng.randint(low=minLen,high=maxLen+1)
	dataBody = rng.rand(dim,seqLen)
	# dataPad = np.zeros((dim,maxLen - seqLen))
	# data = np.hstack((dataBody,dataPad))
	idx = rng.permutation(seqLen)
	query = np.zeros([maxLen, seqLen])
	target = np.zeros([dim, seqLen])
	for id,value in enumerate(idx):
		oneHot = np.zeros([maxLen])
		oneHot[value] = 1
		query[:, id] = oneHot
		target[:, id] = dataBody[:, value]
	trainData.append(dataBody)
	trainQuery.append(query)
	trainTarget.append(target)

for i in range(totalNumTest):
	seqLen = rng.randint(low=minLen,high=maxLen+1)
	dataBody = rng.rand(dim,seqLen)
	# dataPad = np.zeros((dim,maxLen - seqLen))
	# data = np.hstack((dataBody,dataPad))
	idx = rng.permutation(seqLen)
	query = np.zeros([maxLen, seqLen])
	target = np.zeros([dim, seqLen])
	for id,value in enumerate(idx):
		oneHot = np.zeros([maxLen])
		oneHot[value] = 1
		query[:, id] = oneHot
		target[:, id] = dataBody[:, value]
	testData.append(dataBody)
	testQuery.append(query)
	testTarget.append(target)

with open(saveFileTrain, 'wb') as f:
	cPickle.dump(trainData,f)
	cPickle.dump(trainQuery,f)
	cPickle.dump(trainTarget,f)

with open(saveFileTest, 'wb') as f:
	cPickle.dump(testData,f)
	cPickle.dump(testQuery,f)
	cPickle.dump(testTarget,f)


print "dump finished!"

with open(saveFileTrain,'rb') as f:
	temp = cPickle.load(f)
	temp1 = cPickle.load(f)
	temp2 = cPickle.load(f)

print temp[0]
print temp1[0]
print temp2[0]

