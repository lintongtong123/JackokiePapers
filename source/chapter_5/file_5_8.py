#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 9:56
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_5_8.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# ---------------------------------------------------------------------
# Kernel Number Comparing.
# ---------------------------------------------------------------------

import pickle
import numpy as np

import scipy.interpolate as inp
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def accurs_show_3d(num_layer_1, num_layer_2, accurs, cmap, title, path):
    """ Show the accuracy.
    num_layer_1: The coordinate of the first axes.
    num_layer_2: The coordinate of the second axes.
    accurs: The accuracy of different
    """
    fig = plt.figure(dpi=120, figsize=[8, 6])
    ax = Axes3D(fig)

    fit = inp.interp2d(num_layer_1, num_layer_2, accurs)

    x_n = np.linspace(min(num_layer_1), max(num_layer_1), 10240)
    y_n = np.linspace(min(num_layer_2), max(num_layer_2), 10240)

    accurs_n = fit(x_n, y_n)
    ax.plot_surface(x_n, y_n, accurs_n, cmap=cmap)
    ax.set_xlabel('first layer number')
    ax.set_ylabel('second layer number')
    ax.set_zlabel('accuracy')
    plt.title(title)
    # plt.tight_layout()
    fig.savefig(path)


def accurs_show_2d(num_layer_1, num_layer_2, accurs, y_wide, linecolor, title, path):
    """ Show the accuracy.
    num_layer_1: The coordinate of the first axes.
    num_layer_2: The coordinate of the second axes.
    accurs: The accuracy of different
    """

    fig = plt.figure(dpi=200, figsize=[8, 8])
    p_1_labels = []
    p_1 = []
    for i in range(8):
        p_1_labels.append('first_layer_num--%d' % num_layer_1[i])

    for i in range(8):
        p_1.append(plt.plot(num_layer_2, accurs[i], '--',
                            color=linecolor[i],
                            label=p_1_labels[i]))

    plt.ylim(y_wide)
    plt.xticks(num_layer_2)
    plt.xlabel('second_layer_number')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(path)


def number_compare(kernel_numbers):
    """ Plot the kernel numbers comparing...
    kernel_numbers: the data of kernel numbers.
    """
    x = y = [16, 32, 48, 64, 96, 128, 196, 256]
    linecolor = ['gold', 'red', 'springgreen', 'black', 'blue', 'cyan', 'blueviolet', 'magenta']

    accurs = np.ones([8, 8], dtype=float)
    for m in range(8):
        for n in range(8):
            accurs[m][n] = kernel_numbers[(x[m], y[n])]

    accurs_show_3d(x, y, accurs, plt.cm.rainbow, '不同数目卷积核的分类准确率',
                   'E:/JackokiePapers/figures/chapter_5/fig_5_9.png')
    accurs_show_2d(x, y, accurs, [0, 73], linecolor, '不同数目卷积核的分类准确率',
                   'E:/JackokiePapers/figures/chapter_5/fig_5_10.png')
    accurs_show_2d(x, y, accurs, [66, 72], linecolor, '不同数目卷积核的分类准确率',
                   'E:/JackokiePapers/figures/chapter_5/fig_5_11.png')


if __name__ == '__main__':

    kernel_numbers = pickle.load(open('accuracy_kernels_num.pkl', 'rb'))
    number_compare(kernel_numbers)

    print('The program has finished running...')