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

import pickle
import numpy as np
from matplotlib import cm
import scipy.interpolate as inp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go

def layer_show_3D(layers, width, accuracys, title, path):
    """ Show the accuracy with 3D
    layers: The number of layers.
    width: The width of each layers' kernel width.
    accuracys: The accuracys with layers and width
    """
    fig = plt.figure(dpi=120, figsize=(8, 6))
    ax = Axes3D(fig)
    fit = inp.interp2d(layers, width, accuracys)
    y_n = np.linspace(min(layers), max(layers), 5120)
    x_n = np.linspace(min(width), max(width), 5120)
    epoches_n = fit(y_n, x_n)
    surf = ax.plot_surface(y_n, x_n, epoches_n, cmap=cm.rainbow)
    # plt.title(title)
    ax.set_xlabel('layers number')
    ax.set_ylabel('kernel width')
    ax.set_zlabel('accuracy')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.tight_layout()
    plt.savefig(path)


def layer_hot_map(layer, width, accuracys, colormap=None, title=None, path=None):
    """ Plot the hot map of data.

    """
    fig = plt.figure(dpi=120, figsize=(8, 7))
    x_n = np.linspace(min(layer), max(layer), 1001*len(layer))
    y_n = np.linspace(min(width), max(width), 1001*len(width))
    fit = inp.interp2d(layer, width, accuracys)
    accurs_n = fit(x_n, y_n)
    surf = plt.imshow(accurs_n, cmap=colormap)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xticks([])
    plt.yticks([])

    first_str = ''
    second_str = 10 * ' ' + 'Each Layer Kernel Width\n\n'
    for num in layer:
        first_str = first_str + str(num) + 13*' '
    first_str = first_str.strip()
    first_str = first_str + '\n\n' + 8 * ' ' + 'Layers Number'

    for index in range(len(width)):
        cu_width = width[len(width)-index-1]
        second_str = second_str + str(cu_width) + 13*' '
    second_str = second_str.rstrip()
    plt.xlabel(first_str)
    plt.ylabel(second_str)
    plt.tight_layout()
    plt.savefig(path)



def layer_show_mean(layer, width, accurs, y_wide=None, title=None, path=None):
    """ Show the accuracy with different layers and kernel width.
    layers: The number of layers.
    width: The width of each layers' kernel width.
    accurs: The accuracys with layers and width
    """
    fig = plt.figure(dpi=120, figsize=[8, 6])
    plt.plot(layer, accurs.mean(axis=0), '--o', label='mean_accuracy')

    plt.ylim(y_wide)
    plt.xticks(layer)
    plt.xlabel('layers number')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.title(title)
    plt.tight_layout()
    fig.savefig(path)


def layer_show_2D(layer, width, accurs, y_wide, linecolor, title=None, mean_show=False, path=None):
    """ Show the accuracy with different layers and kernel width.
    layers: The number of layers.
    width: The width of each layers' kernel width.
    accurs: The accuracys with layers and width
    """
    fig = plt.figure(dpi=120, figsize=[8, 6])
    p_1_labels = []
    p_1 = []
    for i in range(len(width)):
        p_1_labels.append('kernel_width--%d' % width[i])

    for i in range(len(width)):
        p_1.append(plt.plot(layer, accurs[i], '--',
                            color=linecolor[i],
                            label=p_1_labels[i]))
    if mean_show:
        means = np.mean(accurs, 0)
        plt.plot(layer, means, '-ro', linewidth=2, label='mean_accuray')

    plt.ylim(y_wide)
    plt.xticks(layer)
    plt.xlabel('layers number')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    # plt.title(title)
    fig.savefig(path)


def layer_compare(accuracy_layer_numbers):
    """ Compare the accuracy with layer number changing.
    accuracy_layer_numbers: accuracy dict with layer number changing.
    """
    width = np.linspace(3, 12, 10, dtype=int)
    layer = np.linspace(1, 9, 9, dtype=int)
    accurs = np.ones([10, 9], dtype=float)

    for m in range(10):
        for n in range(9):
            accurs[m][n] = accuracy_layer_numbers[(width[m], layer[n])]

    linecolor = ['gold', 'red', 'green', 'springgreen', 'black', 'blue', 'cyan', 'blueviolet', 'magenta', 'navy']

    layer_show_3D(layer, width, accurs,
                  title='不同数目卷积层和卷积核宽度的分类准确率',
                  path='E:/JackokiePapers/figures/chapter_5/fig_5_1.png')

    layer_hot_map(layer, width, accurs,
                  colormap=cm.rainbow,
                  path='E:/JackokiePapers/figures/chapter_5/fig_5_2.png')

    layer_show_2D(layer, width, accurs, [0, 72],
                  linecolor=linecolor,
                  mean_show=True,
                  path='E:/JackokiePapers/figures/chapter_5/fig_5_4.png')


if __name__ == '__main__':
    data = pickle.load(open('accuracy_layers.pkl', 'rb'))
    layer_compare(data)
    print('The program has finished running...')
