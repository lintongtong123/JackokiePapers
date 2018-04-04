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
    surf = ax.plot_surface(x_n, y_n, accurs_n, cmap=cmap)
    fig.colorbar(surf, shrink=0.5, aspect=5)
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


# def mean_show(x_ticks, accurs, y_wide, linecolor, title, path):
#     """ Show the mean accurs of data."""
#     fig = plt.figure(dpi=200, figsize=[8, 8])
#     p_1_labels = []
#     p_1 = []
#     for i in range(8):
#         p_1_labels.append('first_layer_num--%d' % num_layer_1[i])
#
#     for i in range(8):
#         p_1.append(plt.plot(num_layer_2, accurs[i], '--',
#                             color=linecolor[i],
#                             label=p_1_labels[i]))
#
#     plt.ylim(y_wide)
#     # plt.xticks(num_layer_2)
#     plt.xlabel('second_layer_number')
#     plt.ylabel('accuracy')
#     plt.legend()
#     plt.title(title)
#     plt.tight_layout()
#     fig.savefig(path)


# def mean_accurs(layer_1_1, layer_1_2, layer_2_1, layer_2_2):
#     """ Show the mean accuracy of different layer with."""
#
#     mean_accurs_layer_1 = np.mean(layer_1_1, axis=1)
#     mean_accurs_layer_2 = np.mean(layer_1_1, axis=0)
#     x_ticks = np.linspace(2, 9, num=8, dtype=int)


def kernel_size_heat_map(first_layer_width, second_layer_width, accuracys, colormap=None, title=None, path=None):
    """ Plot the hot map of data.

    """
    fig = plt.figure(dpi=120, figsize=(8, 7))
    y_n = np.linspace(min(second_layer_width), max(second_layer_width), 1001*len(second_layer_width))
    x_n = np.linspace(min(first_layer_width), max(first_layer_width), 1001*len(first_layer_width))
    fit = inp.interp2d(first_layer_width, second_layer_width, accuracys)
    accurs_n = fit(y_n, x_n)
    surf = plt.imshow(accurs_n, cmap=colormap)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xticks([])
    plt.yticks([])

    first_str = ''
    second_str = 10 * ' ' + 'Second Layer Kernel Width\n\n'
    for width in first_layer_width:
        first_str = first_str + str(width) + 16*' '
    first_str = first_str.strip()
    first_str = first_str + '\n\n' + 8 * ' ' + 'First Layer Kernel Width'

    for index in range(len(second_layer_width)):
        width = second_layer_width[len(second_layer_width)-index-1]
        second_str = second_str + str(width) + 16*' '
    second_str = second_str.rstrip()

    plt.ylabel(second_str)
    plt.xlabel(first_str)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)


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

    # mean_accurs(layer_1_1, layer_1_2, layer_2_1, layer_2_2)

    # accurs_show_3d(x, y, layer_1_1, plt.cm.rainbow,
    #                r'1 $\times$ 1 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_15.png')
    #
    # accurs_show_3d(x, y, layer_1_2, plt.cm.rainbow,
    #                r'1 $\times$ 2 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_16.png')
    # accurs_show_3d(x, y, layer_2_1, plt.cm.rainbow,
    #                r'2 $\times 1$ layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_17.png')
    # accurs_show_3d(x, y, layer_2_2, plt.cm.rainbow,
    #                r'2 $\times$ 2 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_18.png')
    #
    # accurs_show_2d(x, y, layer_1_1, [0, 73], linecolor,
    #                r'1 \times 1 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_19.png')
    # accurs_show_2d(x, y, layer_1_2, [0, 73], linecolor,
    #                r'1 \times 2 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_20.png')
    # accurs_show_2d(x, y, layer_2_1, [0, 73], linecolor,
    #                r'2 \times 1 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_21.png')
    # accurs_show_2d(x, y, layer_2_2, [0, 73], linecolor,
    #                r'2 \times 2 layers accuracy',
    #                'E:/JackokiePapers/figures/chapter_5/fig_5_22.png')

    kernel_size_heat_map(x, y, layer_1_1,
                         plt.cm.rainbow,
                         r'1 $\times$ 1 accuracy',
                         'E:/JackokiePapers/figures/chapter_5/fig_5_23.png')
    kernel_size_heat_map(x, y, layer_1_2,
                         plt.cm.rainbow,
                         r'1 $\times$ 2 accuracy',
                         'E:/JackokiePapers/figures/chapter_5/fig_5_24.png')
    kernel_size_heat_map(x, y, layer_2_1,
                         plt.cm.rainbow,
                         r'2 $\times$ 1 accuracy',
                         'E:/JackokiePapers/figures/chapter_5/fig_5_25.png')
    kernel_size_heat_map(x, y, layer_2_2,
                         plt.cm.rainbow,
                         r'2 $\times$ 2 accuracy',
                         'E:/JackokiePapers/figures/chapter_5/fig_5_26.png')

if __name__ == '__main__':

    kernel_sizes = pickle.load(open('accuracy_kernels_size.pkl', 'rb'))
    print(len(kernel_sizes))
    size_compare(kernel_sizes)

    print('The program has finished running...')