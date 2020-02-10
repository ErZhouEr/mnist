import _pickle as cPickle
import gzip
import numpy as np
from chardet import detect
# import matplotlib.pyplot as plt

def load_data(path):
    f=gzip.open(path,'rb')
    # l=f.readline()
    # print(detect(l))  # {'encoding': 'Windows-1252', 'confidence': 0.73, 'language': ''}
    training_data, validation_data, test_data=cPickle.load(f,encoding='unicode-escape')   # unicode-escape为什么就可以
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper(path):
    tr_d, va_d, te_d = load_data(path)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def load_data_wrapper_list(path):
    tr_d, va_d, te_d = load_data(path)
    training_inputs = [list(x) for x in tr_d[0]]
    training_results = [_list_y(y) for y in tr_d[1]]
    training_data = (training_inputs, training_results)
    validation_inputs = [list(x) for x in va_d[0]]
    validation_data = (validation_inputs, list(va_d[1]))
    test_inputs = [list(x) for x in te_d[0]]
    test_data = (test_inputs, list(te_d[1]))
    return (training_data, validation_data, test_data)

def _list_y(y):
    ylst=[0]*10
    ylst[y]=1
    return ylst



def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# def showimg(x):
#     pic=x.reshape(28,28)
#     plt.imshow(pic)

if __name__=='__main__':
    training_data, validation_data, test_data = load_data('mnist.pkl.gz')
