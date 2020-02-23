import data_loader
import time
from neutral_network_matrix import neutral_network_m,neutral_network_m2

train_data, validation_data, test_data=data_loader.load_data_wrapper('mnist.pkl.gz')

# nnmodel_ori=neutral_network_m.NeutralNet([784,100,10])
# nnmodel_ori.train_SGD(train_data,40,10,3.0,test_data)   # 二次损失函数，40个迭代，获得了96.24%的准确率

nnmodel_CrsEntCost=neutral_network_m2.NeutralNet([784,100,10],cost_func=neutral_network_m2.CrossEntropyCost)
nnmodel_CrsEntCost.train_SGD(train_data,40,10,3.0,test_data)  # 交叉伤损失函数，40个迭代，训练的更快了，而且获得了96.48%的准确率


