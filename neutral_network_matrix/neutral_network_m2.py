import numpy as np
import json
import sys


class QuadraticCost(object):
    @staticmethod
    def ori_func(y, label):
        return 0.5 * (label - y) * (label - y)

    @staticmethod
    def d_func(y, label):
        return y - label

    @staticmethod
    def delta(z, y, label):
        return (y - label) * Sigmod_activefunc.sigmod_derivative(z)


class CrossEntropyCost(object):
    @staticmethod
    def ori_func(y, label):
        # np.nan_to_num使用0代替数组x中的nan元素，使用有限的数字代替inf元素，这里的np.sum没什么用吧
        return np.sum(np.nan_to_num(-label * np.log(y) - (1 - label) * np.log(1 - y)))

    @staticmethod
    def d_func(y, label):
        return (y - label) / (y * (1 - y))

    @staticmethod
    def delta(z, y, label):
        return (y - label)


class Sigmod_activefunc(object):

    @staticmethod
    def sigmod(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmod_derivative(z):
        a = Sigmod_activefunc.sigmod(z)
        return a * (1 - a)


class NeutralNet(object):
    def __init__(self, layer_num, cost_func=QuadraticCost, init_func='ori_paraminit'):
        self.layer_count = len(layer_num)
        self.layer_num = layer_num
        self.sizes = layer_num
        self.cost_func = cost_func
        if init_func == 'ori_paraminit':
            self.ori_paraminit()
        elif init_func == 'paraminit':
            self.paraminit()

    def ori_paraminit(self):
        self.bias = [np.random.randn(y, 1) for y in self.layer_num[1:]]  # 输入层没有bias，np.random.randn(y,1)返回y行1列的正态分布样本
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layer_num[:-1], self.layer_num[1:])]

    def paraminit(self):
        self.bias = [np.random.randn(y, 1) for y in self.layer_num[1:]]  # 偏置没什么影响
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_num[:-1], self.layer_num[1:])]
        # weight变成标准差为1/sqrt(n)

    def feedforward(self, a):
        for b, w in zip(self.bias, self.weights):
            a = self.sigmod(np.dot(w, a) + b)  # numpy 的各种运算法则还不是很清楚
        return a

    def train_SGD(self, train_data, epochs, mini_batch_size, eta, lamd, test_data=None,
                  evaluation_data=None, monitor=False):
        '''
        理解随机梯度下降，从下面代码可以看出，每次用一个大小为k的mini_batch的样本来更新参数，相当于对整个train_data分组后进行循环，
        如果与批量梯度下降的超参数迭代次数epochs相同，其实应该相当于没有效率的优化，但是区别在于同样的epochs，随机梯度下降相当于迭代了
        epochs*（n_train/k）次，即SGD的epochs其实是epoch*（n_train/k）,n_train/k一般很大，所以实际上不需要这么多，一般其epochs
        远小于批量梯度下降，所以效率得到了大幅优化
        '''
        # if test_data:
        #     n_test = len(test_data)
        n_train = len(train_data)
        evaluation_accs,evaluation_costs=[],[]
        train_accs,train_costs=[],[]
        for i in range(epochs):
            mini_batchs = [train_data[k:k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta, lamd, n_train)
            if monitor:
                eval_acc=self.cal_acc(evaluation_data,is_traindata=False)
                eval_cost=self.cal_cost(evaluation_data,is_traindata=False)
                train_acc=self.cal_acc(train_data,is_traindata=True)
                train_cost=self.cal_cost(train_data,is_traindata=True)
                evaluation_accs.append(eval_acc)
                evaluation_costs.append(eval_cost)
                train_accs.append(train_acc)
                train_costs.append(train_cost)
                print(f'Epoch{i}:aval_acc:{eval_acc},train_acc:{train_acc}\n eval_cost:{eval_cost},train_cost:{train_cost}')
            print(f'----------Epoch{i}:Completed----------')
        return evaluation_accs,evaluation_costs,train_accs,train_costs

    def update_mini_batch(self, mini_batch, eta, lamd, n):
        b_tmp = [np.zeros(b.shape) for b in self.bias]
        w_tmp = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            b_tmp = [nb + dnb for nb, dnb in zip(b_tmp, delta_b)]
            w_tmp = [nw + dnw for nw, dnw in zip(w_tmp, delta_w)]
        # 随机梯度下降是mini_batch中每个样本梯度的平均，注意理解这个平均,lamd*eta*w/n实现了L2正则
        self.bias = [b - eta / len(mini_batch) * nb for b, nb in zip(self.bias, b_tmp)]
        self.weights = [w - eta / len(mini_batch) * nw - lamd * eta * w / n for w, nw in zip(self.weights, w_tmp)]

    def backprop(self, x, y):
        b_tmp = [np.zeros(b.shape) for b in self.bias]
        w_tmp = [np.zeros(w.shape) for w in self.weights]
        # 前向，与feedforward函数的区别是，每一层的输出a和加权输入z需要记录下来
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmod(z)
            activations.append(activation)
        # 后向，根据前向记录的结果得到delta
        # delta = self.cost_func.d_func(activations[-1], y) * self.sigmod_derivative(zs[-1])
        # 出现了0/0的情况，bug，故直接调用损失函数的delta函数了
        delta = self.cost_func.delta(zs[-1], activations[-1], y)
        b_tmp[-1] = delta
        w_tmp[-1] = np.dot(delta, activations[-2].transpose())  # activations[-2]层的输出，就是输出层的输入x
        # 对np.array各个纬度的转变需要弄清楚
        # 接下来对隐含层
        for l in range(2, self.layer_count):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmod_derivative(z)
            b_tmp[-l] = delta
            w_tmp[-l] = np.dot(delta, activations[-l - 1].transpose())
        return b_tmp, w_tmp

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]  # 取输出节点最大值的索引为预测值
        return sum([int(x == y) for x, y in test_results])

    def cost_derivative(self, y_output, y):
        return y_output - y

    def sigmod(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmod_derivative(self, z):
        a = self.sigmod(z)
        return a * (1 - a)

    def cal_acc(self, data,is_traindata=False):
        # train_data的label是被做过向量化的，跟evaluation和test的y有区别
        N=len(data)
        if is_traindata:
            result=[(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            result=[(np.argmax(self.feedforward(x)),y) for x,y in data]
        return sum([int(x==y) for x,y in result])/N

    def cal_cost(self, data,is_traindata=False):
        N=len(data)
        if is_traindata:
            result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            result = [(np.argmax(self.feedforward(x)), y) for x, y in data]
        return sum([self.cost_func.ori_func(y,label) for y,label in result])/N

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.bias],
                "cost": str(self.cost_func.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = NeutralNet(data["sizes"], cost_func=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
