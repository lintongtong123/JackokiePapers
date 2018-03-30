#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:43
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_5_7.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# ---------------------------------------------------------------------
# Layer Comparing.
# ---------------------------------------------------------------------

import numpy as np

import scipy.interpolate as inp
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def kernel_epoches():
    """ Graph the kernel number change with epoches."""
    x = y = [16, 32, 48, 64, 96, 128, 196, 256]
    epoches = [[57.130, 29.688, 26.368, 26.563, 23.243, 25.587, 15.235, 13.282],
               [32.618, 26.661, 25.489, 26.563, 28.126, 44.435, 26.075, 23.829],
               [41.700, 35.938, 23.243, 35.450, 29.688, 26.759, 27.540, 27.735],
               [38.087, 31.251, 31.349, 38.087, 17.481, 25.294, 32.130, 15.821],
               [41.993, 24.122, 17.384, 22.657, 35.157, 29.396, 17.091, 16.310],
               [24.024, 35.841, 33.009, 28.224, 31.739, 16.310, 22.071, 23.829],
               [41.993, 35.255, 28.224, 17.970, 19.728, 20.802, 18.556, 19.923],
               [33.106, 27.149, 20.509, 32.423, 21.583, 15.919, 15.138, 16.993]]

    fig = plt.figure(dpi=120)
    ax = Axes3D(fig)

    fit = inp.interp2d(x, y, epoches)

    x_n = y_n = np.linspace(16, 256, 5120)
    epoches_n = fit(x_n, y_n)
    ax.plot_surface(x_n, y_n, epoches_n, cmap='magma')
    plt.show()


if __name__ == '__main__':
    data = pickle.load(open('accuracy_kernels_num'))
    print('The program has finished running...')