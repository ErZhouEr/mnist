#!usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author:ZHOUCE
@file: util_funcs.py
@time: 2020/03/11
"""
import numpy as np


def cal_output_size(input_size, filter_size, stride, padding):
	return (input_size + 2 * padding - filter_size) / stride + 1


def pad(input_array, pad_size):
	'''
	为数组增加Zero padding，自动适配输入为2D和3D的情况
	'''
	if pad_size == 0:
		return input_array
	else:
		if input_array.ndim == 2:
			in_width = input_array.shape[1]
			in_height = input_array.shape[0]
			padded_array = np.zeros((in_height + 2 * pad_size, in_width + 2 * pad_size))  # 这里要求搞清楚array的行列索引情况
			padded_array[pad_size:in_height + pad_size, pad_size:in_width + pad_size] = input_array
			return padded_array
		elif input_array.ndim == 3:
			in_width = input_array.shape[2]
			in_height = input_array.shape[1]
			in_depth = input_array.shape[0]
			padded_array = np.zeros((in_depth, in_height + 2 * pad_size, in_width + 2 * pad_size))
			padded_array[:, pad_size:in_height + 2 * pad_size, in_width + 2 * pad_size] = input_array
			return padded_array


def convolution(input_array, kernel_array, stride, output_array, bias):
	'''
	卷积计算，自动适应2D或3D的输入矩阵
	'''
	output_width = output_array.shape[1]
	output_height = output_array.shape[0]
	kernel_width = kernel_array.shape[1]
	kernel_height = kernel_array.shape[0]
	for w in output_width:
		for h in output_height:
			if input_array.ndim == 2:
				output_array[h, w] = input_array[stride * h:stride * h + kernel_height,
				                     stride * w:stride * w + kernel_width] * kernel_array.sum() + bias
				# *乘是按元素乘,np.matmul是矩阵乘, 索引的前面是h，后面是w
			elif input_array.ndim == 3:
				output_array[h, w] = input_array[:, stride * h:stride * h + kernel_height,
				                     stride * w:stride * w + kernel_width] * kernel_array.sum() + bias
				# 加上深度后的变化就是矩阵在表示是需要多加一个维度，全部元素sum之后作为输出的一个节点


def element_wise_op(array,operator):
	'''
	按元素运算
	'''
	for i in np.nditer(array,op_flags=['readwrite']):
		i[...]=operator(i)      # i[...]是要干什么？
