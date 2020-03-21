#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:ZHOUCE
@file: recurrentNN.py
@time: 2020/03/20
"""

import numpy as np
from cnn import util_funcs

class RecNNLayer(object):
	def __init__(self,input_width,state_width,activator,learning_rate):
		'''
		循环层, 只包括输出到状态节点，不包括状态节点到输出节点，所以只有权重U、W，没有V
		:param input_width: 只有width，难道实现的是一维的输入吗
		'''
		self.input_width=input_width
		self.state_width=state_width
		self.activator=activator
		self.learning_rate=learning_rate
		self.time=0
		self.state_lst=[]  # 用来存放各个时间的state_array
		self.state_lst.append(np.zeros((state_width,1)))  # np.zeros((state_width,1)) 是初始化的time0的state_array
		self.U=np.zeros((state_width,input_width))        # 初始化U
		self.W=np.zeros((state_width,state_width))        # 初始化W


	def forward(self,input_array):
		state=np.dot(self.U,input_array)+np.dot(self.W,self.state_lst[-1])
		self.output=util_funcs.element_wise_op(state,self.activator)
		self.state_lst.append(state)
		self.time+=1

	def backward(self,sensitivity_array,activator):
		pass