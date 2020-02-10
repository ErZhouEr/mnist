from neutral_network_ori.node import Node

class Layer(object):

    def __init__(self,layer_index,node_num):
        self.layer_index=layer_index
        self.node_num=node_num
        self.nodes=[]
        for i in range(node_num):
            self.nodes.append(Node(layer_index,i))


    def cal_output(self):
        for node in self.nodes:
            node.cal_output()

    def set_output(self,data_set):     #输入层的输入节点需要此函数
        for node,data in zip(self.nodes,data_set):
            node.set_output(data)
