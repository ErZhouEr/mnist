#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:ZHOUCE
@file: filter.py
@time: 2020/03/10
"""
import numpy as np


class Filter(object):
	def __init__(self, width, height, channel):
		self.width = width
		self.height = height
		self.channel = channel
		self.weights = np.random.randn(channel, height, width)
		self.bias = 0
		self.weight_grads = np.zeros(self.weights.shape)
		self.bias_grad = 0

	def get_weights(self):
		return self.weights

	def get_bias(self):
		return self.bias

	def update(self, rate):
		self.weights -= self.weight_grads * rate
		self.bias -= self.bias_grad * rate

	def __repr__(self):
		return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))
