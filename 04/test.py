import numpy as np
from read_mnist import *

dataset = list(read_dataset("training","."))

dataset_plain = list(read_plain_dataset("training","."))
labels, features = read_first_labels(dataset = 'training', path = '.', n=1)


print(len(dataset))
label, pixels = dataset[0]
print(label)
print(pixels.shape)
#print(pixels)
show(pixels)

print('plain')

n = 3

m = len(dataset_plain)
limited_labels = [];
features = [];
for i in range (0,m):
    label, feature = dataset_plain[i]
    if( label < n ):
        limited_labels = limited_labels + [label]
        features = features + [feature]

print(len(labels))
print(len(features))
print(features[0])

label, feature = dataset_plain[0]
print(labels)
print(feature.shape)
print(feature)

r = np.where(label<n)
ans = np.where(label==1)[0:]


print('bye')

