#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/03/31

import numpy as np


class LinearRegression:
    """
    拟合一元线性回归模型

    Parameters
    ----------
    x : shape 为(样本个数,)的 numpy.array
        只有一个属性的数据集

    y : shape 为(样本个数,)的 numpy.array
        标记空间

    Returns
    -------
    self : 返回 self 的实例.
    """
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x, y):
        self.w = np.sum(y * (x - np.mean(x))) / (np.sum(x**2) - (1/x.size) * (np.sum(x))**2)
        self.b = (1 / x.size) * np.sum(y - self.w * x)
        return self

    def predict(self, x):
        """
        使用该线性模型进行预测

        Parameters
        ----------
        x : 数值 或 shape 为(样本个数,)的 numpy.array
            属性值

        Returns
        -------
        C : 返回预测值
        """
        return self.w * x + self.b
