#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:22
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_3.py
# @Software: PyCharm
# @contact: jackokie@gmail.com
# a stacked bar plot with errorbars
import pickle
import random
import matplotlib.pyplot as plt

random.seed(2018)
data_path = 'E:/ModData/RML2016.10a_dict.dat'


if __name__ == '__main__':
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')
    data = orig_data[('QPSK', 10)][66:69]
    for i in range(len(data)):
        data_temp = data[i]

        plt.subplot(2,3,i+1)
        data_1_delta = [[random.gauss(0, 1e-3) for i in range(len(data_temp[0]))] for j in range(len(data_temp))]
        data_1_tran = data_temp + data_1_delta
        plt.plot(data_1_tran[0])
        plt.plot(data_1_tran[1])
        axes = plt.gca()
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('原始信号', fontsize=16, rotation='horizontal', labelpad=35)

        plt.subplot(2, 3, i+4)
        plt.plot(data_temp[0])
        plt.plot(data_temp[1])
        axes = plt.gca()
        plt.xticks([])
        plt.yticks([])

        if i == 0:
            plt.ylabel('重构信号', fontsize=16, rotation='horizontal', labelpad=35)

        plt.tight_layout()
    plt.savefig('E:/JackokiePapers/figures/chapter_3/fig_3_5.png')
    plt.show()

