import neutral_network_ori.active_func as active_func
import neutral_network_ori.loss_func as loss_func

class Node(object):

    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.bias = 0.0
        self.up_conns = []
        self.down_conns = []
        self.net = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.active_func = active_func
        self.loss_func = loss_func

    def set_output(self,data):   #输入节点需要此函数
        self.output=data

    def cal_output(self):
        self.net = sum([conn.weight * conn.up_node.output + self.bias for conn in self.up_conns])
        self.output = self.active_func.orifunc(self.net)

    def cal_output_delta(self,label):
        self.cal_output()
        self.delta = self.loss_func.d_func(label,self.output) * self.active_func.d_func(self.net)

    def cal_hidden_delta(self):
        self.delta = self.active_func.d_func(self.net) * sum(
            [conn.weight * conn.down_node.delta for conn in self.down_conns])

    def update_bias(self, rate):
        self.bias += self.delta * rate

    def add_upconns(self, conn):
        self.up_conns.append(conn)

    def add_downconns(self, conn):
        self.down_conns.append(conn)

    def __str__(self):
        return f'节点{self.layer_index}-{self.node_index}'
