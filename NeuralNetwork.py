import numpy as np
from math import exp

#create framework for a three layer Neural Network
class NeuralNetwork():

    def __init__(self,numInputs,numHiddenNeurons,numOutputs):
        self.numInputs = numInputs
        self.numHiddenNeurons = numHiddenNeurons
        self.numOutputs = numOutputs
        self.weightsInputToHidden = self.__initializeWeights(numInputs,numHiddenNeurons)
        self.weightsHiddenToOutput = self.__initializeWeights(numHiddenNeurons,numOutputs)
                

    def __initializeWeights(self,numOfSources,numOfDestinations):                  

        rows = numOfDestinations
        cols = numOfSources + 1 #add 1 for bias
        weights = []

        for row in range(rows):

            currentRow = [np.random.random() for col in range(cols)]
            weights.append(currentRow)

        #return a matrix with rows containing the input to each destination layer     
        return np.array(weights)


    '''
    Train the network through a feed-forward, backpropagation(stochastic gradient descent) algorithm:
    1: feed the data forward through the network
        A: inputs * weights + bias -> hidden layer
        B: compress data through sigmoid function
        C: hidden layer * weights + bias -> output
        D: smush data through sigmoid function
    2: make prediction from the output
    3: calculate the error between the actual classification and the prediction
    4: propagate the error backwards to adjust the weights accordingly
        A: calculate the gradients for the weights between each layer
        B: update weight matricies such that W = W + G
    '''
    def train(self,learningRate,trainingData):

        #trainingData must be a list where the first element is an array containing the inputs and
        #the second element is an array which pertains to the associated label or expected output        
        for trainer in trainingData:            
            #create a vector for the actual output
            targetVector = np.reshape(trainer[1],(self.numOutputs,1))

            #create an Nx1 vector with the input data            
            inputVector = np.append(trainer[0],[1])
            inputVector = np.reshape(inputVector,(self.numInputs+1,1))

            #calculate hidden layer vector
            hiddenVector = np.dot(self.weightsInputToHidden,inputVector)
            #run each value through sigmoid filter
            hiddenVector = [self.__Sigmoid(i) for i in hiddenVector]    
            hiddenVector.append(1) #add bias       
            hiddenVector = np.reshape(hiddenVector,(self.numHiddenNeurons+1,1))
                   
            #calculate output vector
            outputVector = np.dot(self.weightsHiddenToOutput,hiddenVector)
            outputVector = [self.__Sigmoid(i) for i in outputVector]            
            outputVector = np.reshape(outputVector,(self.numOutputs,1))            
            #calculate the error
            errorVectorOutput = np.subtract(targetVector,outputVector)
            

            #Backpropagation starts here
            #calculate the new weights for the edges between the Hidden and Output layers
            gradientVectorHiddenToOutput = self.__calculateGradient(learningRate,errorVectorOutput,hiddenVector,outputVector)
            # self.weightsHiddenToOutput = np.add(self.weightsHiddenToOutput,gradientVectorHiddenToOutput)

            #calculate the hidden layer errors as the transpose of the weight matrix between
            #Hidden and Output layers multiplied by the output error vector
            errorVectorHidden = np.dot(np.transpose(self.weightsHiddenToOutput),errorVectorOutput)
            # print(errorVectorHidden)
            # print(hiddenVector)
            # errorVectorHidden = errorVectorHidden[:len(errorVectorHidden)-1]
            # hiddenVector = hiddenVector[:len(hiddenVector)-1]
            gradientVectorInputToHidden = self.__calculateGradient(learningRate,errorVectorHidden,inputVector[:2],hiddenVector)
            self.weightsInputToHidden = np.add(self.weightsInputToHidden,np.transpose(gradientVectorInputToHidden))
            self.weightsHiddenToOutput = np.add(self.weightsHiddenToOutput,gradientVectorHiddenToOutput)

    # dW = alpha * Error * d(sigmoid(x)) * Input
    def __calculateGradient(self,learningRate,errorVector,sourceVector,destinationVector):

        gradientVector = np.ones((len(destinationVector),1))
        gradientVector = np.subtract(gradientVector,destinationVector)
        gradientVector = gradientVector * gradientVector
        gradientVector = learningRate * errorVector * gradientVector
        gradientVector = np.dot(gradientVector,np.transpose(sourceVector))

        return gradientVector


    def __Sigmoid(self,x):

        return 1 / (1 + exp(-x))

    def __dSigmoid(self,x):

        return self.__Sigmoid(x) * (1 - self.__Sigmoid(x))


    '''
    To make a prediction:
    1: create an input vector
    2: calculate the hidden layer vector as weights * inputs
    3: smush data through sigmoid
    4: calculate the outpt layer vector as weights * hidden
    5: smush data through sigmoid
    6: return the output vector
    '''
    def predict(self,data):

        inputVector = np.array(data)
        inputVector = np.append(inputVector,[1]) #add bias
        inputVector = np.reshape(inputVector,(self.numInputs+1,1))

        hiddenVector = np.dot(self.weightsInputToHidden,inputVector)
        hiddenVector = [self.__Sigmoid(i) for i in hiddenVector]
        hiddenVector.append(1)
        hiddenVector = np.reshape(hiddenVector,(self.numHiddenNeurons+1,1))

        outputVector = np.dot(self.weightsHiddenToOutput,hiddenVector)
        outputVector = [self.__Sigmoid(i) for i in outputVector]
        # outputVector = np.reshape(outputVector,(self.numOutputs,1))

        return outputVector