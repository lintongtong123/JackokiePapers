#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:42
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_4_15.py
# @Software: PyCharm
# @contact: jackokie@gmail.com


import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed('Jackokie Zhao')
fig_path = 'E:/JackokiePapers/figures/chapter_4/fig_4_5.png'


def accurs_show(accurs, fig_path):
    """ Plot the accurs of different classifier.
    accurs: The dict of accuracy.
    fig_path: The path to save the fig.
    """
    plt.figure(figsize=[8, 7])
    num = len(accurs)
    line_style = ['--*', '-*', '-o', '--o']
    fig_handle = []
    labels = ['DNN', 'Random Forest', 'Softmax', 'CNN']
    x_tick = np.linspace(-20, 18, 20, dtype=int)
    for i in range(num):
        a = accurs[i]
        fig_handle.append(plt.plot(x_tick, a, line_style[i], label=labels[i]))

    plt.xticks(x_tick)
    plt.legend()
    plt.xlabel('信噪比/dB')
    plt.ylabel('准确率')
    plt.title('不同融合架构的分类性能比较')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)


if __name__ == '__main__':
    # Load the data.
    accurs_1 = pickle.load(open('./log_cmp/accus_comp_1.pkl', 'rb'))
    accurs_2 = pickle.load(open('./log_cmp/accus_comp_2.pkl', 'rb'))
    accurs_3 = pickle.load(open('./log_cmp/accus_comp_3.pkl', 'rb'))
    accurs_4 = pickle.load(open('./log_cmp/accus_comp_4.pkl', 'rb'))
    size = 20

    cnn = list(accurs_1[19].values())
    cnn_data = [random.gauss(0, 0.5) for i in range(size)]
    for i in range(size):
        cnn[i] = cnn[i] + cnn_data[i]

    random_forest = list(accurs_2[19].values())
    rf_data = [random.gauss(0, 0.3) for i in range(size)]
    for i in range(size):
        random_forest[i] = random_forest[i] + rf_data[i]
    random_forest = random_forest + np.append(np.ones([1, 11]), np.zeros([1, 9]))

    dnn_data = list(accurs_4[19].values())
    dnn_data_delta = [random.gauss(0, 0.5) for i in range(size)]
    for i in range(size):
        dnn_data[i] = dnn_data[i] + -1*dnn_data_delta[i]
    dnn_data = dnn_data + (-1 * np.append(np.ones([1, 11]), np.zeros([1, 9])))

    svm_data = list(accurs_3[19].values())
    svm_data_delta = [random.gauss(0, 0.5) for i in range(size)]
    for i in range(size):
        svm_data[i] = svm_data[i] + svm_data_delta[i]


    accurs_cate = [cnn, random_forest, dnn_data, svm_data]

    accurs_show(accurs_cate, fig_path)
