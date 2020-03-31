#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/03/31

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LinearRegression import LinearRegression


def runplt():
    plt.figure()
    plt.title("Pizza price plotted against diameter")
    plt.xlabel('Diameter')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    return plt


# 加载数据
pizza = pd.read_csv("data/pizza.csv", index_col='Id')
dia = pizza.loc[:,'Diameter'].values
price = pizza.loc[:,'Price'].values

# 创建并拟合模型
model = LinearRegression()
model.fit(dia, price)

x2 = np.array([0., 25.])  # 取两个预测值
y2 = model.predict(x2)  # 进行预测
print(y2)  # 查看预测值

runplt()
plt.plot(dia, price, 'k.')
plt.plot(x2, y2, 'g-')  # 画出拟合曲线
plt.show()
