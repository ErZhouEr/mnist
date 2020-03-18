#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:l
@file: convolution.py
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
		self.output_width = util_funcs.cal_output_size(input_width, filter_width, stride, padding)
		self.output_height = util_funcs.cal_output_size(input_height, filter_height, stride,padding)
		self.output_array = np.zeros((filter_num, self.output_height, self.output_width))
		# 这个维度的计算需要注意, 卷积后输出的维度是filter的数量，channel作为深度不影响输出的数量
		self.filters = []
		for i in range(filter_num):  # filter的宽、高、深度即channel
			self.filters.append(Filter(filter_width, filter_height, channel_num))

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

	def backward(self, input_array, sensitivity_array, activator):
		'''
		反向传播，一共三个步骤：前向、误差项反向传播、计算梯度
		:param input_array: 用于前向
		:param sensitivity_array: 用于反向传播
		:param activator: 用于反向传播
		:return: 计算上一层的误差项、参数的梯度
		'''
		self.forward(input_array)
		self.bp_sensitivity_map(sensitivity_array, activator)
		self.bp_gradient(sensitivity_array)

	def update_w_b(self):
		'''
		更新所有卷积核的权重、偏置
		'''
		for f in self.filters:
			f.update(self.rate)

	def bp_sensitivity_map(self, sensitivity_array, activator):
		'''
		向上一层传递误差项
		:param sensitivity_array:本层误差项
		:param activator:上一层的激活函数，这些本层上层的关系要随时根据链式法则判断
		'''
		# 1 卷积步长不一定为1，对于不为1的，要对原始sensitivity map进行扩展
		sensitivity_map = self.expand_sensitivity_map(sensitivity_array)

		# 2 要对sensitivity_map进行pad，因为上一层边缘的节点的误差传播仅与sensitivity_map的边缘有关，要想进行卷积，就要补pad
		pad = (self.input_width + self.filter_width - 1 - sensitivity_map.shape[2]) / 2
		# 这个计算原理：补0后的sensitivity_map要能够作为输入得到真正输入array的width
		# 即解方程：(sen_map_width - filter_width + 2*pad)/stride+1 = input_width
		padded_map = util_funcs.pad(sensitivity_map, pad)

		# 3 针对每个filter，进行误差传递, 传递的结果是得到与上一层（即input）相同维度的误差项
		self.delta_array = np.zeros_like(self.input_array)  # 卷积层的误差项，input_array的形状：[channel,height,width]
		# 多个filter，需要将每个filter对应的误差项相加
		for f in range(self.filter_num):
			fliped_filter = np.rot90(self.filters[f].get_weights(), 2)
			tmp = np.zeros((self.input_height, self.input_width))
			# 多个channel，需要将filter的channel与上一层的channel对应
			for d in tmp.shape[0]:
				# filter 的个数对应着输出的层数，所以padded_map[f]
				util_funcs.convolution(padded_map[f], fliped_filter[d], 1, tmp[d], 0)
			self.delta_array += tmp
		# 最后得到的delta是维度[channel,height,width]的array

		# 4 按位乘上一层激活函数的导数项
		derivative_func = activator.d_func
		derivative_array = np.array(self.input_array)
		util_funcs.element_wise_op(derivative_array, derivative_func)
		self.delta_array *= derivative_array

	def bp_gradient(self, sensitivity_array):
		'''
		计算每层权重梯度
		:param sensitivity_array: 本层误差项
		'''
		# 1 卷积步长不一定为1，对于不为1的，要对原始sensitivity map进行扩展
		expanded_map = self.expand_sensitivity_map(sensitivity_array)

		# 2 计算 w 的梯度，本层误差项与本层输入（上层输出）进行卷积，结果得到与filter维度相同的梯度array
		for f in range(self.filter_num):
			filter = self.filters[f]
			for d in range(filter.channel):
				# 每个不同channel的误差项是一样的，想想前向时候输出是怎么计算的就理解了
				# 为什么用padded_input_arr？因为padded_input_arr与filter卷积得到输出的维度，所以
				# padded_input_arr与输出卷积能得到filter的维度
				util_funcs.convolution(self.padded_input_arr[d], expanded_map[f], 1, filter.weight_grads[d], 0)
			# 每个filter一个bias
			filter.bias_grad = expanded_map[f].sum()

	def expand_sensitivity_map(self, sensitivity_array):
		'''
		针对可能的步长不为1的情况进行处理，一般对误差项矩阵进行扩展，把被跳过的步长补0
		:param sensitivity_array: 原始误差项矩阵
		'''
		depth = sensitivity_array.shape[0]
		expand_width = util_funcs.cal_output_size(self.input_width, self.filter_width, stride=1, padding=self.padding)
		expand_height = util_funcs.cal_output_size(self.input_height, self.filter_height, stride=1,padding=self.padding)
		expand_map = np.zeros([depth, expand_height, expand_width])
		for i in self.input_height:
			for j in self.input_width:
				ii = i * self.stride
				jj = j * self.stride
				expand_map[:, ii, jj] = sensitivity_array[:, i, j]  # depth会一一对应上吗, 经过试验，会的
		return expand_map
