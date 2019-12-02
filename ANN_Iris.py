import numpy as np
import NN2 as nn
import csv
import random

#a function that returns the index of the row for the corresponding species being tested
def UpdateTable(species):

    if species == "Iris-setosa":
        dim = 0
    elif species == "Iris-versicolor":
        dim = 1
    else:
        dim = 2

    return dim


def main():

	#retrieve data from CSV file
	with open('iris_dataset.csv') as dataFile:
		data = np.array(list(csv.reader(dataFile)))

	#create lists for training and testing
	trainingSet = []
	testingSet = []
	count = 0
	for iris in data:
		
		dataVector = [float(i) for i in iris[:4]]
		if iris[4] == 'Iris-setosa':
			classificationVector = [1,0,0]
		elif iris[4] == 'Iris-versicolor':
			classificationVector = [0,1,0]
		else:
			classificationVector = [0,0,1]
		sortData = [dataVector,classificationVector]
		if count % 2 == 0:
			trainingSet.append(sortData)
		else:
			testingSet.append(sortData)
		count = count + 1

	learningRate = 0.01

	trainingSetPlus = trainingSet[:]
	for i in range(10000):
		random.shuffle(trainingSetPlus)
		trainingSetPlus = trainingSetPlus + trainingSet[:]
		random.shuffle(trainingSetPlus)

	network = nn.NeuralNetwork(4,3,3)
	network.train(learningRate,trainingSet)

	#build confusion matrix
	'''
	+------------+---------+------------+-----------+
	|            | Setosa  | Versicolor | Virginica |
	+------------+---------+------------+-----------+
	| Setosa     |  TP(S)  |   E(S,Ve)  |  E(S,Vi)  |
	| Versicolor | E(Ve,S) |   TP(Ve)   |  E(Ve,Vi) |
	| Virginica  | E(Vi,S) |   E(Vi,Ve) |   TP(Vi)  |
	+------------|---------+------------+-----------|
	'''
	confusionMatrix = np.zeros(shape=(3,3)) #creates a 2D array (3x3 table) containing all zeros

	for test in testingSet:

		prediction = network.predict(test[0])
		print(prediction)
		#find classification
		highestIndex = 0
		for index in range(1,len(prediction)):
			if prediction[index] > prediction[highestIndex]:
				highestIndex = index
		
		if highestIndex == 0:
			prediction = 'Iris-setosa'
		elif highestIndex == 1:
			prediction = 'Iris-versicolor'
		else:
			prediction = 'Iris-virginica'

		#make a prediction for the current test case
		classification = test[1].index(1)
		row = UpdateTable(classification)
		col = UpdateTable(prediction)
		confusionMatrix[row,col] = confusionMatrix[row,col] + 1 #increment corresponding cell in the matrix

	print(confusionMatrix)


if __name__ == "__main__":
	main()

