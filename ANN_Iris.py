import numpy as np
from collections import OrderedDict


# data = {(0,0):0,
#         (0,1):1,
#         (1,0):1,
#         (1,1):0
#         }

# dataKeys = list(data.keys())
# dataValues = list(data.values())

# print (data[(0,1)])
# print()
# print(dataKeys)
# print(dataKeys[0])
# print()
# print(dataValues)
# print(dataValues[0])

# print()
# print()
# dataItems = list(data.items())
# print(dataItems)
# print(dataItems[0][0])
# test = [dataKeys[0][i] for i in range(2)]
# print(test)


data = OrderedDict((0,0):0,
        (0,1):1,
        (1,0):1,
        (1,1):0
        )

print(data)