class Connection(object):

    def __init__(self, up_node, down_node):
        self.up_node = up_node
        self.down_node = down_node
        self.weight = 0.0
        self.gradient = 0.0

    def cal_gradient(self):
        self.gradient = self.down_node.delta * self.up_node.output

    def update_weight(self, rate):
        self.cal_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        return f'{self.up_node.layer_index}-{self.up_node.node_index} --> {self.down_node.layer_index}-{self.down_node.node_index}'
