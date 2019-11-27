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
def logicalNAND(x,y):
    
    if x + y < 2:
        return 1
    else:
        return 0


def logicalOR(x,y):

    if x + y > 0:
        return 1
    else:
        return 0


def logicalAND(x,y):

    if x + y > 1:
        return 1
    else:
        return 0


#XOR(x,y) = AND(OR(x,y),NAND(x,y))
def logicalXOR(x,y):

    prediction = logicalAND(logicalNAND(x,y),logicalOR(x,y))

    return prediction


def main():

    inputs = [(0,0),(0,1),(1,0),(1,1)]

    for test in inputs:
        out = logicalXOR(test[0],test[1])
        print(out)

if __name__=='__main__':
    main()

