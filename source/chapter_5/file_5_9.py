#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 9:56
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_5_9.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# ---------------------------------------------------------------------
# Kernel Size Comparing.
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


def mean_show(x_ticks, accurs, y_wide, linecolor, title, path):
    """ Show the mean accurs of data."""
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

def mean_accurs(layer_1_1, layer_1_2, layer_2_1, layer_2_2):
    """ Show the mean accuracy of different layer with."""

    mean_accurs_layer_1 = np.mean(layer_1_1, axis=1)
    mean_accurs_layer_2 = np.mean(layer_1_1, axis=0)
    x_ticks = np.linspace(2, 9, num=8, dtype=int)


def size_compare(kernel_sizes):
    """ Plot the kernel numbers comparing...
    kernel_numbers: the data of kernel numbers.
    """
    layer_1_1 = np.ones([8, 8], dtype=float)
    layer_1_2 = np.ones([8, 8], dtype=float)
    layer_2_1 = np.ones([8, 8], dtype=float)
    layer_2_2 = np.ones([8, 8], dtype=float)

    x = y = np.linspace(2, 9, num=8, dtype=int)
    linecolor = ['gold', 'red', 'springgreen', 'black', 'blue', 'cyan', 'blueviolet', 'magenta']
    for m in x:
        for n in y:
            layer_1_1[m-2, n-2] = kernel_sizes[(1, 1, m, n)]
            layer_1_2[m-2, n-2] = kernel_sizes[(1, 2, m, n)]
            layer_2_1[m-2, n-2] = kernel_sizes[(2, 1, m, n)]
            layer_2_2[m-2, n-2] = kernel_sizes[(2, 2, m, n)]

    mean_accurs(layer_1_1, layer_1_2, layer_2_1, layer_2_2)

    # accurs_show_3d(x, y, layer_1_1, plt.cm.rainbow,
    #                r'1 \times 1 layers accuracy',
    #                './layer_1_1.png')
    # accurs_show_3d(x, y, layer_1_2, plt.cm.rainbow,
    #                r'1 \times 2 layers accuracy',
    #                './layer_1_2.png')
    # accurs_show_3d(x, y, layer_2_1, plt.cm.rainbow,
    #                r'2 \times 1 layers accuracy',
    #                './layer_2_1.png')
    # accurs_show_3d(x, y, layer_2_2, plt.cm.rainbow,
    #                r'2 \times 2 layers accuracy',
    #                './layer_2_2.png')
    #
    # accurs_show_2d(x, y, layer_1_1, [0, 73], linecolor,
    #                r'1 \times 1 layers accuracy',
    #                './kernel_size_1_1.png')
    # accurs_show_2d(x, y, layer_1_2, [0, 73], linecolor,
    #                r'1 \times 2 layers accuracy',
    #                './kernel_size_1_2.png')
    # accurs_show_2d(x, y, layer_2_1, [0, 73], linecolor,
    #                r'2 \times 1 layers accuracy',
    #                './kernel_size_2_1.png')
    # accurs_show_2d(x, y, layer_2_2, [0, 73], linecolor,
    #                r'2 \times 2 layers accuracy',
    #                './kernel_size_2_2.png')


if __name__ == '__main__':

    kernel_sizes = pickle.load(open('accuracy_kernels_size.pkl', 'rb'))
    size_compare(kernel_sizes)

    print('The program has finished running...')