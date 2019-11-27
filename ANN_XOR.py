'''
Implement a multi-layer feedforward back-propagation algorithm which perfroms the XOR function.

XOR:
+-----+-----+------+-----+------+-----+
|  x  |  y  |  OR  | AND | NAND | XOR |
+-----+-----+------+-----+------+-----+
|  0  |  0  |   0  |  0  |   1  |  0  |
|  1  |  0  |   1  |  0  |   1  |  1  |
|  0  |  1  |   1  |  0  |   1  |  1  |
|  1  |  1  |   1  |  1  |   0  |  0  |
+-----+-----+------+-----+------+-----+

XOR is the "exclusive OR" logic function, it operates by taking the OR between two inputs and outputting 1
when the inputs are distinct and one input is 1. It can be reduced to the combination of OR and NOT-AND.

To represent the XOR function as an ANN, we will create a three layer network in which the input layer has two inputs
'''

import NeuralNetwork as nn
import numpy as np

def main():

    # dataSet = [[[0,0],[1,0]],
    #             [[0,1],[0,1]],
    #             [[1,0],[0,1]],
    #             [[1,1],[1,0]]
    #             ]

    dataSet = [[[0,0],[0]],
                [[0,1],[1]],
                [[1,0],[1]],
                [[1,1],[0]]
                ]

    learningRate = 0.01
    numberOfTrainingSamples = 100000
    trainingData = [dataSet[np.random.randint(0,4,size=None,dtype=int)] for i in range(numberOfTrainingSamples)]

    network = nn.NeuralNetwork(2,2,1)
    network.train(learningRate,trainingData)
    
    for test in dataSet:
        prediction = network.predict(test[0])
        print(f"Input: {test[0]} | Target: {test[1]} | Predicted: {prediction}")


if __name__ == "__main__":
    main()

