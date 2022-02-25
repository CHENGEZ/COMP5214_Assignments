from mlxtend.data import loadlocal_mnist

def load_train_data():
    return loadlocal_mnist("./train-images.idx3-ubyte","./train-labels.idx1-ubyte")

def load_test_data():
    return loadlocal_mnist("./t10k-images.idx3-ubyte","./t10k-labels.idx1-ubyte")
