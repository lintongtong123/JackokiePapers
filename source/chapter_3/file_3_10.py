#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:23
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_10.py
# @Software: PyCharm
# @contact: jackokie@gmail.com


# Kernel Size Comparing.
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inp
from mpl_toolkits.mplot3d import Axes3D

def kernels_show():
    fig = plt.figure()
    ax = Axes3D(fig)
    y = x = [i+1 for i in range(8)]
    temp = np.random.random((8, 8))
    fit = inp.interp2d(x, y ,temp, kind='cubic')
    y_n = x_n = np.linspace(2, 9, 100)
    temp_n = fit(x_n, y_n)
    ax.plot_surface(x_n, y_n, temp_n, rstride=1, cstride=1, cmap='Purples')

    plt.show()


if __name__ == '__main__':
    kernels_show()
    print()
