from neutral_network_ori.layer import Layer
from neutral_network_ori.connections import Connections
from neutral_network_ori.connection import Connection
import neutral_network_ori.loss_func


class NeutralNet(object):

    def __init__(self, layer_num):  # 初始化一个全联接网络，所以不需要指定connections，layer_num是一个list，元素代表每层node数
        self.layers = []
        layer_count = len(layer_num)
        for i in range(layer_count):
            self.layers.append(Layer(i, layer_num[i]))  # 初始化layer，给各layer填充节点

        self.connections = Connections()
        for i in range(layer_count - 1):  # 初始化connections，全连接网络
            connections = [Connection(up_node, down_node) for up_node in self.layers[i].nodes
                           for down_node in self.layers[i + 1].nodes]
            for conn in connections:
                self.connections.add_conn(conn)
                conn.up_node.add_downconns(conn)
                conn.down_node.add_upconns(conn)

    def train(self, train_data, labels, rate, N):
        for i in range(N):
            for x, y in zip(train_data, labels):
                self.train_one_sample(x, y, rate)

    def train_one_sample(self, x, y, rate):
        self.predict(x)
        self._cal_delta(y)
        self._update_weight(rate)

    def predict(self, x):
        self.layers[0].set_output(data_set=x)  # 根据样本设置输入层
        for layer in self.layers[1:]:  # 计算各层输出
            layer.cal_output()
        return [node.output for node in self.layers[-1].nodes]  # 返回输出层的结果

    def _cal_delta(self, y):
        for node, label in zip(self.layers[-1].nodes, y):
            node.cal_output_delta(label)
        for layer in self.layers[-2::-1]:  # 反向传播误差
            for node in layer.nodes:
                node.cal_hidden_delta()

    def _update_weight(self, rate):
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.update_bias(rate)          # 更新节点bias
                for conn in node.down_conns:
                    conn.update_weight(rate)    # 更新连接weight

    def _cal_gradient(self):                     # 计算网络每层的梯度，留待检查梯度计算的正确性
        for layer in self.layers[1:]:
            for node in layer.nodes:
                for conn in node.up_conns:
                    conn.cal_gradient()

    def get_gradient(self, x, y):
        self.predict(x)
        self._cal_delta(y)
        self._cal_gradient()

    def gradient_check(network, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''
        # 计算网络误差
        network_error = neutral_network.loss_func.orifunc
        # 获取网络在当前样本下每个连接的梯度
        network.get_gradient(sample_feature, sample_label)
        # 对每个权重做梯度检查
        for conn in network.connections.connections:
            # 获取指定连接的梯度
            actual_gradient = conn.get_gradient()
            # 增加一个很小的值，计算网络的误差
            epsilon = 0.0001
            conn.weight += epsilon
            error1 = network_error(network.predict(sample_feature), sample_label)
            # 减去一个很小的值，计算网络的误差
            conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
            error2 = network_error(network.predict(sample_feature), sample_label)
            # 根据式6计算期望的梯度值
            expected_gradient = (error2 - error1) / (2 * epsilon)
            # 打印
            print('expected gradient: \t%f\nactual gradient: \t%f' % (
                expected_gradient, actual_gradient))

    def dump(self):
        pass


if __name__ == '__main__':
    neutral_network_ori = NeutralNet([10, 3, 1])


