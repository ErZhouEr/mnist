import numpy as np


class QuadraticCost(object):
    @staticmethod
    def ori_func(y, label):
        return 0.5 * (label - y) * (label - y)

    @staticmethod
    def d_func(y, label):
        return y - label

    @staticmethod
    def delta(z,y,label):
        return (y-label)*Sigmod_activefunc.sigmod_derivative(z)


class CrossEntropyCost(object):
    @staticmethod
    def ori_func(y, label):
        # np.nan_to_num使用0代替数组x中的nan元素，使用有限的数字代替inf元素，这里的np.sum没什么用吧
        return np.sum(np.nan_to_num(-label * np.log(y) - (1 - label) * np.log(1 - y)))

    @staticmethod
    def d_func(y, label):
        return (y-label)/(y*(1-y))

    @staticmethod
    def delta(z,y,label):
        return (y-label)


class Sigmod_activefunc(object):

    @staticmethod
    def sigmod(z):
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def sigmod_derivative(z):
        a = Sigmod_activefunc.sigmod(z)
        return a * (1 - a)


class NeutralNet(object):
    def __init__(self, layer_num, cost_func=QuadraticCost):
        self.layer_count = len(layer_num)
        self.sizes = layer_num
        self.bias = [np.random.randn(y, 1) for y in layer_num[1:]]  # 输入层没有bias，np.random.randn(y,1)返回y行1列的正态分布样本
        # self.weights = [[[np.random.randn(layer_num[layer + 1], 1)] for _ in range(layer_num[layer])] for layer in
        #                 range(self.layer_count - 1)]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_num[:-1], layer_num[1:])]
        self.cost_func=cost_func

    def feedforward(self, a):
        for b, w in zip(self.bias, self.weights):
            a = self.sigmod(np.dot(w, a) + b)  # numpy 的各种运算法则还不是很清楚
        return a

    def train_SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        理解随机梯度下降，从下面代码可以看出，每次用一个大小为k的mini_batch的样本来更新参数，相当于对整个train_data分组后进行循环，
        如果与批量梯度下降的超参数迭代次数epochs相同，其实应该相当于没有效率的优化，但是区别在于同样的epochs，随即梯度下降相当于迭代了
        epochs＊（n_train/k）次，实际上不需要这么多，一半其epochs远小与批量梯度下降，所以效率得到了大幅优化
        '''
        # if test_data:
        #     n_test = len(test_data)
        n_train = len(train_data)
        for i in range(epochs):
            mini_batchs = [train_data[k:k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_test = len(test_data)
                print(f'Epoch{i}:{self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch{i}:Completed')

    def update_mini_batch(self, mini_batch, eta):
        b_tmp = [np.zeros(b.shape) for b in self.bias]
        w_tmp = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            b_tmp = [nb + dnb for nb, dnb in zip(b_tmp, delta_b)]
            w_tmp = [nw + dnw for nw, dnw in zip(w_tmp, delta_w)]
        # 随机梯度下降是mini_batch中每个样本梯度的平均，注意理解这个平均
        self.bias = [b - eta / len(mini_batch) * nb for b, nb in zip(self.bias, b_tmp)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, w_tmp)]

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
        delta=self.cost_func.delta(zs[-1],activations[-1], y)
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
