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
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D


def accurs_show_3d(num_layer_1, num_layer_2, accurs, cmap, title, path):
    """ Show the accuracy.
    num_layer_1: The coordinate of the first axes.
    num_layer_2: The coordinate of the second axes.
    accurs: The accuracy of different
    """
    fig = plt.figure(dpi=120, figsize=[8, 7])
    ax = Axes3D(fig)

    fit = inp.interp2d(num_layer_1, num_layer_2, accurs)

    x_n = np.linspace(min(num_layer_1), max(num_layer_1), 10240)
    y_n = np.linspace(min(num_layer_2), max(num_layer_2), 10240)

    accurs_n = fit(x_n, y_n)
    surf = ax.plot_surface(y_n, x_n, accurs_n, cmap=cmap)
    ax.set_xlabel('Second Layer Number')
    ax.set_ylabel('First Layer Number')
    ax.set_zlabel('accuracy')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.title(title)
    fig.savefig(path)

def kernel_num_hot_map(layers, accuracys, colormap=None, title=None, path=None):
    """ Plot the hot map of data.

    """
    fig = plt.figure(dpi=120, figsize=(8, 7))
    y_n = np.linspace(min(layers), max(layers), 1001*len(layers))
    x_n = np.linspace(min(layers), max(layers), 1001*len(layers))
    fit = inp.interp2d(layers, layers, accuracys)
    accurs_n = fit(x_n, y_n)
    surf = plt.imshow(accurs_n, cmap=colormap)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('                                        First Layer Kernel Number                    '
               '                         \n\n'
               '256             196              128               96               64               48'
               '              32              16')

    plt.xlabel('16             32              48               64               96               128'
               '              196              256\n\n'
               '                        Second Layer Kernel Number                              ')
    plt.tight_layout()
    plt.savefig(path)


def accurs_show_2d(num_layer_1, num_layer_2, accurs, y_wide, axis_x, linecolor, title=None, mean_show=False, path=None):
    """ Show the accuracy.
    num_layer_1: The coordinate of the first axes.
    num_layer_2: The coordinate of the second axes.
    accurs: The accuracy of different
    """

    fig = plt.figure(dpi=120, figsize=[8, 6])
    p_1_labels = []
    p_1 = []
    if axis_x == 0:
        for i in range(len(num_layer_1)):
            p_1_labels.append('first_layer_num--%d' % num_layer_1[i])

        for i in range(len(num_layer_1)):
            p_1.append(plt.plot(num_layer_1, accurs[i], '--',
                                color=linecolor[i],
                                label=p_1_labels[i]))
        if mean_show:
            means = np.mean(accurs, axis=axis_x)
            plt.plot(num_layer_1, means, '-o', linewidth=2, label='mean_accuray')
        plt.ylim(y_wide)
        plt.xticks(num_layer_1)
        plt.xlabel('second_layer_number')
        plt.ylabel('accuracy')
    else:
        for i in range(len(num_layer_2)):
            p_1_labels.append('second_layer_num--%d' % num_layer_2[i])

        for i in range(len(num_layer_2)):
            p_1.append(plt.plot(num_layer_1, accurs[:, i], '--',
                                color=linecolor[i],
                                label=p_1_labels[i]))
        if mean_show:
            means = np.mean(accurs, axis=axis_x)
            plt.plot(num_layer_2, means, '-ro', linewidth=2, label='mean_accuray')
        plt.ylim(y_wide)
        plt.xticks(num_layer_2)
        plt.xlabel('first_layer_number')
        plt.ylabel('accuracy')

    plt.legend()
    if not title is None:
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

    kernel_num_hot_map(x, accurs, colormap=cm.rainbow,
                  path='E:/JackokiePapers/figures/chapter_5/fig_5_10.png')

    accurs_show_2d(x, y, accurs, [0, 73],
                   axis_x=1,
                   linecolor=linecolor,
                   mean_show=False,
                   path='E:/JackokiePapers/figures/chapter_5/fig_5_11.png')

    accurs_show_2d(x, y, accurs, [66, 72],
                   axis_x=1,
                   linecolor=linecolor,
                   mean_show=True,
                   path='E:/JackokiePapers/figures/chapter_5/fig_5_12.png')

    accurs_show_2d(x, y, accurs, [0, 73],
                   axis_x=0,
                   linecolor=linecolor,
                   mean_show=False,
                   path='E:/JackokiePapers/figures/chapter_5/fig_5_13.png')

    accurs_show_2d(x, y, accurs, [66, 72],
                   axis_x=0,
                   linecolor=linecolor,
                   mean_show=True,
                   path='E:/JackokiePapers/figures/chapter_5/fig_5_14.png')


if __name__ == '__main__':

    kernel_numbers = pickle.load(open('accuracy_kernels_num.pkl', 'rb'))
    # print(kernel_numbers)
    number_compare(kernel_numbers)

    print('The program has finished running...')