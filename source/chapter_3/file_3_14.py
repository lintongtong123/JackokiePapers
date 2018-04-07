#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:24
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_14.py
# @Software: PyCharm
# @contact: jackokie@gmail.com


# Classify time.
import matplotlib.pyplot as plt
import numpy as np
name = 'E:/JackokiePapers/figures/chapter_3/fig_3_13.png'
data = [1.2e1, 0.6e-1, 0.3e-1, 2.2e-2, 6.5e-3, 1.2e-3]
height = np.log10(data)
labels = ['KNN', 'SVM', 'CAE-CNN', 'DNN', 'RF', 'DTREE']
plt.yscale('log')
plt.bar(range(len(data)), data, fc='gray', tick_label=labels)
plt.xlabel('分类器')
plt.ylabel('分类时间(s)')
plt.tight_layout()
plt.savefig(name)
