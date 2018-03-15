#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 23:07
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_2_2.py
# @Software: PyCharm
# @contact: jackokie@gmail.com


import numpy as np
import matplotlib.pyplot as plt
# matplotlib.rc('font', size=30)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x = np.arange(-20, 20, 0.1)
y = 1/ (1 + np.exp(-x))

plt.axhline(1, color='k', linestyle='--')
plt.xlim([-20, 20])
fig.add_axes()
plt.plot(x, y, 'r')
plt.xlabel('$z$', fontsize=12)
plt.ylabel("$f(z)$", rotation='horizontal',fontsize=12)

plt.show()

# fig.savefig('/home/scl/Documents/JackokiePapers/figures/chapter_2/fig_2_2.png')
fig.savefig('E:/JackokiePapers/figures/chapter_2/fig_2_3.png')
