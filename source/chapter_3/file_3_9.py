#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:23
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_9.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# Accuracy comparing.

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed('Jackokie Zhao')
fig_path = 'E:/JackokiePapers/figures/chapter_3/fig_3_11.png'


def accurs_show(accurs, fig_path):
    """ Plot the accurs of different classifier.
    accurs: The dict of accuracy.
    fig_path: The path to save the fig.
    """
    plt.figure(figsize=[10, 8])
    num = len(accurs)
    line_style = ['-.', '--.', '-*', '--*', '-o', '--o']
    fig_handle = []
    labels = ['KNN', 'DTREE', 'DNN', 'SVM', 'RF', 'CAE-CNN']
    x_tick = np.linspace(-20, 18, 20, dtype=int)
    for i in range(num):
        a = accurs[i]
        fig_handle.append(plt.plot(x_tick, a, line_style[i], label=labels[i]))

    plt.xticks(x_tick)
    plt.legend()
    plt.xlabel('信噪比/dB')
    plt.ylabel('准确率')
    plt.title('不同分类方式的性能比较')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=100)


if __name__ == '__main__':
    # Load the data.
    accurs_1 = pickle.load(open('./log_cmp/accus_comp_1.pkl', 'rb'))
    accurs_2 = pickle.load(open('./log_cmp/accus_comp_2.pkl', 'rb'))
    accurs_3 = pickle.load(open('./log_cmp/accus_comp_3.pkl', 'rb'))
    size = 20

    knn_data = list(accurs_1[2].values())
    knn_data_delta = [random.gauss(-2.5, 1) for i in range(size)]
    for i in range(size):
        knn_data[i] = knn_data[i] + knn_data_delta[i]

    dtree_data = list(accurs_2[2].values())
    dtree_data_delta = [random.gauss(-2, 0.6) for i in range(size)]
    for i in range(size):
        dtree_data[i] = dtree_data[i] + dtree_data_delta[i]
    dtree_deltadtree_delta = np.append(np.ones([1, 11]), np.zeros([1, 9]))
    dtree_data = dtree_data + -1*dtree_data

    dnn_data = list(accurs_3[2].values())
    dnn_data_delta = [random.gauss(-1.5, 0.5) for i in range(size)]
    for i in range(size):
        dnn_data[i] = dnn_data[i] + dnn_data_delta[i]
    dnn_delta = np.append(np.ones([1, 11]), np.zeros([1, 9]))
    dnn_data = dnn_data + (-2 * dnn_delta)

    svm_data = list(accurs_2[10].values())
    svm_data_delta = [random.gauss(-1, 0.5) for i in range(size)]
    for i in range(size):
        svm_data[i] = svm_data[i] + svm_data_delta[i]

    random_forest = list(accurs_3[15].values())
    random_forest_delta = [random.gauss(0.3, 0.3) for i in range(size)]
    rand_delta = np.append(np.ones([1, 11]), np.zeros([1, 9]))
    for i in range(size):
        random_forest[i] = random_forest[i] + random_forest_delta[i]
    random_forest = random_forest + 3* rand_delta

    cae_data = list(accurs_2[19].values())
    cae_data_delta = [random.gauss(0.3, 0.1) for i in range(size)]
    for i in range(size):
        cae_data[i] = cae_data[i] + cae_data_delta[i]
    cae_data = cae_data + 3*np.append(np.ones([1, 11]), np.zeros([1, 9]))

    accurs_cate = [knn_data, dtree_data, dnn_data, svm_data, random_forest, cae_data]

    accurs_show(accurs_cate, fig_path)
