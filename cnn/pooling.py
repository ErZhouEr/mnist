import numpy as np
from cnn import util_funcs

class PoolingLayer(object):
    '''
    池化层
    '''
    def __init__(self,input_width,input_height,channel_num,filter_width,filter_height,stride,activitor):
        self.input_width=input_width
        self.input_height=input_height
        self.channel=channel_num
        self.filter_width=filter_width
        self.filter_height=filter_height
        self.stride=stride
        self.output_width=util_funcs.cal_output_size(input_width,filter_width,stride,0)
        self.output_height = util_funcs.cal_output_size(input_height, filter_height, stride, 0)
        self.output_array=np.zeros(channel_num,self.output_height,self.output_width)
        self.activitor=activitor

    def forward(self,input_array,activitor):
        '''
        池化层前向
        :param input_array: 输入向量
        :param activitor: 预留一个activitor，可以扩展为多种池化方法
        '''
        pass

    def backward(self,input_array, sensitivity_array, activator):
        pass
