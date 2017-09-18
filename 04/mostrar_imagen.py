import numpy as np
from read_mnist import *

dataset = list(read_dataset("training","."))

print(len(dataset))
label, pixels = dataset[0]
print(label)
print(pixels.shape)
print(pixels)
show(pixels)

print('bye')
