#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:24
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_13.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# Train time.
import matplotlib.pyplot as plt
import numpy as np
name = 'E:/JackokiePapers/figures/chapter_3/fig_3_12.png'
data = [1.8e3, 1.6e3, 1.2e3, 5.2e2, 100, 20]
height = np.log10(data)
labels = ['SVM', 'DNN', 'CAE-CNN', 'XgBoost', 'DTREE', 'KNN']
plt.yscale('log')
plt.bar(range(len(data)), data, fc='gray', tick_label=labels)
plt.xlabel('分类器')
plt.ylabel('训练时间(s)')
plt.tight_layout()
plt.savefig(name)
