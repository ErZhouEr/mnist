#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:l
@file: convolution_nn.py
@time: 2020/03/05
"""

import numpy as np
from cnn.filter import Filter
from cnn.activators import ReluActivator
import cnn.util_funcs as util_funcs


class ConvLayer(object):
	'''
	卷积层
	'''
	def __init__(self, input_width, input_height, channel_num, filter_width, filter_height, filter_num,
	             padding, stride, activator, rate):  # filter 的深度跟输入的channel_num是一样的，所以只定义一个filter_num就行了
		self.input_width = input_width
		self.input_height = input_height
		self.channel_num = channel_num
		self.filter_width = filter_width
		self.filter_height = filter_height
		self.filter_num = filter_num
		self.padding = padding
		self.stride = stride
		self.activator = activator
		self.rate = rate
		self.output_width = util_funcs.cal_output_size(self.input_width, self.filter_width, self.stride, self.padding)
		self.output_height = util_funcs.cal_output_size(self.input_height, self.filter_height, self.stride,
		                                                self.padding)
		self.output_array = np.zeros((self.filter_num, self.output_height, self.output_width))
		# 这个维度的计算需要注意, 卷积后输出的维度是filter的数量，channel作为深度不影响输出的数量
		self.filters = []
		for i in range(self.filter_num):  # filter的宽、高、深度即channel
			self.filters.append(Filter(self.filter_width, self.filter_height, self.channel_num))

	def forward(self, input_array):
		'''
		前向，计算卷积层的输出
		'''
		self.input_array = input_array
		self.padded_input_arr = util_funcs.pad(input_array, self.padding)
		for i in range(self.filter_num):
			filter = self.filters[i]
			util_funcs.convolution(self.padded_input_arr, filter.get_weights(), self.stride, self.output_array[i],
			                       filter.get_bias())
		util_funcs.element_wise_op(self.output_array, self.activator)


	def backward(self,input_array,sensitivity_array,activator):
		'''
		反向传播，一共三个步骤：前向、误差项反向传播、计算梯度
		:param input_array: 用于前向
		:param sensitivity_array: 用于反向传播
		:param activator: 用于反向传播
		:return: 计算上一层的误差项、参数的梯度
		'''
		self.forward(input_array)
		self.bp_sensitivity_map(sensitivity_array,activator)
		self.bp_gradient(sensitivity_array)

	def update_w_b(self):
		'''
		更新所有卷积核的权重、偏置
		'''
		for f in self.filters:
			f.update(self.rate)

	def bp_sensitivity_map(self,sensitivity_array,activator):
		'''
		向上一层传递误差项
		:param sensitivity_array:本层误差项
		:param activator:上一层的激活函数，这些本层上层的关系要随时根据链式法则判断
		'''
		# 1 卷积步长不一定为1，对于不为1的，要对原始sensitivity map进行扩展
		sensitivity_map=self.expand_sensitivity_map(sensitivity_array)
		# 2 要对sensitivity_map进行pad，因为上一层边缘的节点的误差传播仅与sensitivity_map的边缘有关，要想进行卷积，就要补pad
		pad=(self.input_width+self.filter_width-1-sensitivity_map.shape[2])/2
		# 这个计算原理：补0后的sensitivity_map要能够作为输入得到真正输入array的width
		# 即解方程：(sen_map_width - filter_width + 2*pad)/stride+1 = input_width
		padded_map=util_funcs.pad(sensitivity_map,pad)
		# 3 针对每个filter，进行误差传递
		delta_array=np.zeros_like(self.input_array)
		for filter in self.filters:
			fliped_filter=np.rot90(filter.get_weights(),2)
			util_funcs.convolution(padded_map,fliped_filter,1,tmp,0)




	def bp_gradient(self,sensitivity_array):
		'''
		反向传播计算梯度
		:param sensitivity_array:
		:return:
		'''
		pass

	def expand_sensitivity_map(self,sensitivity_array):
		'''
		针对可能的步长不为1的情况进行处理，一般对误差项矩阵进行扩展，把被跳过的步长补0
		:param sensitivity_array: 原始误差项矩阵
		'''
		depth=sensitivity_array.shape[0]
		expand_width=util_funcs.cal_output_size(self.input_width,self.filter_width,stride=1,padding=self.padding)
		expand_height=util_funcs.cal_output_size(self.input_height,self.filter_height,stride=1,padding=self.padding)
		expand_map=np.zeros([depth,expand_height,expand_width])
		for i in self.input_height:
			for j in self.input_width:
				ii=i*self.stride
				jj=j*self.stride
				expand_map[:,ii,jj]=sensitivity_array[:,i,j]   # depth会一一对应上吗, 经过试验，会的
		return expand_map

