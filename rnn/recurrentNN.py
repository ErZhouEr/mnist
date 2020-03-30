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
		:param input_width: 一个时刻的输入向量xt的size，比如一个词向量的size
		'''
		self.input_width=input_width   # 输入向量的size
		self.state_width=state_width   # 隐含层的size
		self.activator=activator
		self.learning_rate=learning_rate
		self.time=0
		self.input_lst=[]  # 用来存放各个时间的input_array
		self.state_lst=[]  # 用来存放各个时间的state_array
		self.state_lst.append(np.zeros((state_width,1)))  # np.zeros((state_width,1)) 是初始化的time0的state_array
		self.U=np.random.uniform(-1e-4,1e-4,(state_width,input_width))        # 初始化U
		self.W=np.random.uniform(-1e-4,1e-4,(state_width,state_width))        # 初始化W


	def forward(self,input_array):
		'''
		实现一个时刻的前向，即只是输入序列中一个xt的前向，对于完整的RNN层的前向，需要对一个序列中所有的input_array进行前向
		:param input_array: 输入序列中一个xt
		'''
		self.input_lst.append(input_array)
		state=np.dot(self.U,input_array)+np.dot(self.W,self.state_lst[-1])
		util_funcs.element_wise_op(state,self.activator)
		self.state_lst.append(state)
		self.time+=1

	def backward(self,sensitivity_array,activator):
		'''
		BPTT算法
		分为两个方向，一个是输入方向，属于普通的full_connect，区别是有很多时刻；一个是序列方向，特点是每个时刻都受到之前所有时刻的影响
		:param sensitivity_array: 本层误差项
		:param activator:
		:return:
		'''
		# 按我的理解，要先计算delta延时间线的横向传播，得到每个t的隐含层delta，然后根据每个t的隐含层delta纵向计算得到输入层的delta
		self.bp_sensitivity_W(sensitivity_array, activator)
		self.bp_sensitivity_U(activator)

		self.cal_gradient_W()
		self.cal_gradient_U()


	def update(self):
		self.U -= self.learning_rate*self.gradient_u
		self.W-=self.learning_rate*self.gradient_w

	def bp_sensitivity_W(self,sensitivity_array,activator):
		self.delta_lst_w=[]
		for i in range(self.time):
			self.delta_lst_w.append(np.zeros(self.state_width,1))    # 理解误差项的维度
		self.delta_lst_w.append(sensitivity_array)   # 这样，sensitivity_array就成了time+1时刻的误差项了，机智
		for k in range(self.time-1,0,-1):            # 反向传播，反过来循环
			self.bp_sensitivity_W_k(k,activator)

	def bp_sensitivity_W_k(self,k,activator):
		state=self.state_lst[k].copy()
		util_funcs.element_wise_op(state,activator.d)
		self.delta_lst_w[k]=np.matmul(np.matmul(self.delta_lst_u[k + 1].T, self.W), np.diag(state, 0)).T
		# [n,1].T -> [1,n] -> [1,n] dot [n,n] -> [1,n] -> [1,n] dot [n,n] -> [1,n] -> [1,n].T -> [n,1]

	def bp_sensitivity_U(self,activator):
		self.delta_lst_u=[]
		for i in range(self.time):
			self.delta_lst_u.append(np.zeros(self.input_width,1))    # 初始化
		for k in range(self.time):
			self.delta_lst_u.append(self.bp_sensitivity_U_k(k, activator))

	def bp_sensitivity_U_k(self,k,activator):
		input_array = self.input_lst[k].copy()
		util_funcs.element_wise_op(input_array,activator.d)
		# 隐含层delta delta_lst_w[k] 作为输入层后一层的delta，往输入层传播
		self.delta_lst_u.append(np.matmul(np.matmul(self.delta_lst_w[k].T,self.U),np.diag(input_array,0)).T)
		# [n,1].T -> [1,n] -> [1,n] dot [n,m] -> [1,m] -> [1,m] dot [m,m] -> [1,m] -> [1,m].T -> [m,1]


	def cal_gradient_W(self):
		gradient_lst_w=[]
		for k in range(1,self.time+1):
			gradient_lst_w.append(np.matmul(self.delta_lst_w[k],self.state_lst[k-1]).T)
		self.gradient_w=sum(gradient_lst_w)


	def cal_gradient_U(self):
		gradient_lst_u = []
		for k in range(1, self.time + 1):
			gradient_lst_u.append(np.matmul(self.delta_lst_u[k], self.input_lst[k - 1]).T)
		self.gradient_u = sum(gradient_lst_u)