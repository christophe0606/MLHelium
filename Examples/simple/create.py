import os
import sys
import inspect

import torch
import numpy as np

sys.path.insert(0, '../../scripts')

from export import serialize_tensors,read_tensors

# The two tensor that must be exported for use in C
x = np.array([[0.8145, 0.3953, 0.4753],
        [0.7439, 0.6556, 0.7681],
        [0.0457, 0.2006, 0.1127],
        [0.1944, 0.9207, 0.7807],
        [0.0394, 0.2478, 0.3634]])

y = np.array([[1],
        [2],
        [3]],dtype=np.float16)
#print(x.shape)
#print(x.size)

# The tensors are serialized to a file
# This file can be used as it is (loaded into memory of the board)
# or it can be converted to a C array.
# Tensors are aligned in the buffer on a 8 bytes boundary by default.
# The buffer on board where the content will be loaded must also be
# aligned.
with open("test.dat","wb") as f:
    serialize_tensors(f,[x,y])

# Re-read the network. Shape / strides are not saved into the output file.
# It is to check that the data has been exported correctly
with open("test.dat","rb") as f:
    res = read_tensors(f)

print(res)
