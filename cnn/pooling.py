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
        for d in range(self.channel):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    self.output_array[d,h,w]=input_array[d,h*self.stride:h*self.stride+self.filter_height,
                                             w*self.stride:w*self.stride+self.filter_width].max()

    def backward(self,input_array, sensitivity_array, activator):
        '''
        池化层反向传播
        :param input_array: 输入向量
        :param sensitivity_array: 本层误差项向量
        '''
        self.delta_array=np.zeros_like(input_array)
        for d in range(self.channel):
            for h in range(self.input_height):
                for w in range(self.input_width):
                    k,l=util_funcs.get_max_index(input_array[d, h:h + self.stride, w:w + self.stride])
                    self.delta_array[d,h*self.stride+k,w*self.stride+l]=sensitivity_array[d,h,w]
