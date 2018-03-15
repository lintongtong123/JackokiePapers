#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 22:51
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_2_1.py
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

x = np.arange(-5, 10, 0.1)
y = (x>0)*x

plt.xlim([-5, 12])
fig.add_axes()
plt.plot(x, y, 'r')
plt.xlabel('$z$', fontsize=12)
plt.ylabel("$f(z)$", rotation='horizontal',fontsize=12)

plt.show()

# fig.savefig('/home/scl/Documents/JackokiePapers/figures/chapter_2/fig_2_2.png')
fig.savefig('E:/JackokiePapers/figures/chapter_2/fig_2_2.png')
