import data_loader
import time
from neutral_network_matrix import neutral_network_m

train_data, validation_data, test_data=data_loader.load_data_wrapper('mnist.pkl.gz')
nnmodel=neutral_network_m.NeutralNet([784,30,10])
nnmodel.train_SGD(train_data,10,100,1,test_data)
