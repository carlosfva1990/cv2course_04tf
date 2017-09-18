import os
import struct
import numpy as np

# Lee los features como imagenes 2D
def read_dataset(dataset = 'training', path = '.'):
    if dataset is 'training':
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is 'testing':
        fname_img = os.path.join(path, 't10k-images.idx3-ubte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        print('dataset must be "testing" or "training"')
        return None

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

# Lee solo las imagenes que tienen label menor que n
# features normalizadas y one hot encoded labels
def read_first_labels(dataset = 'training', path = '.', n=1):
    dataset_plain = list(read_plain_dataset("training","."))

    m = int(len(dataset_plain))
    limited_labels = [];
    features = [];
    for i in range (0,m):
        label, feature = dataset_plain[i]
        if( label < n ):
            one_hot = [0.0] * n
            one_hot[label] = 1.0
            limited_labels = limited_labels + [one_hot]
            features = features + [feature*(1/255)]

    return limited_labels, features

# Lee lasfeatures como un vector 1D
def read_plain_dataset(dataset = 'training', path = '.'):
    if dataset is 'training':
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is 'testing':
        fname_img = os.path.join(path, 't10k-images.idx3-ubte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        print('dataset must be "testing" or "training"')
        return None

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


# muestra la imagen
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
