#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:ZHOUCE
@file: activators.py
@time: 2020/03/10
"""
import numpy as np


class ReluActivator(object):
	@staticmethod
	def ori_func(x):
		return max(0,x)

	@staticmethod
	def d_func(y):
		return 1 if y>0 else 0

class SigmodActivator(object):
	@staticmethod
	def ori_func(x):
		return 1/(1+np.exp(-x))

	@staticmethod
	def d_func(y):
		return y*(1-y)


class IdentityActivator(object):
	@staticmethod
	def ori_func(x):
		return x

	@staticmethod
	def d_func(y):
		return 1


class TanhActivator(object):
	@staticmethod
	def ori_func(x):
		return

	@staticmethod
	def d_func(y):
		return 1-y*y