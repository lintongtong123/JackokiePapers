#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:18
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_1.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# data_set visualization

import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
random.seed(2018)


def get_sample(data_path):
    """Get the sample to show.
    Parameters:
        data_path: The data in the system.
    Returns:
        samples: The samples to show.
        cate: The samples' category.
    """
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')
    mode_snr = list(orig_data.keys())
    mods, snrs = [sorted(list(set(x[i] for x in mode_snr))) for i in [0, 1]]
    index = [random.randint(0, 1000) for i in range(len(mods))]
    snr = 18
    samples = [orig_data[(mods[i], snr)][index[i]] for i in range(11)]
    return samples, mods

def plot_samples(samples, mods):
    """ Plot the samples.
    Parameters:
        samples: The samples to be plot.
        mods: The Modulation of samples.
    """

    num = len(samples)
    for i in range(num):

        plt.subplot(4, 3, i+1)
        p1, = plt.plot(samples[i][0], 'b', linestyle='--')
        p2, = plt.plot(samples[i][1], 'r', linestyle='-.')
        plt.hlines(0, 0, 128)

        axes = plt.gca()
        axes.set_ylim([-0.022, 0.022])
        axes.set_xlim([0, 128])

        plt.xticks([])
        plt.yticks([-0.02, 0.02])


        plt.legend([p1, p2], ['Real', 'Complex'], loc='upper right', fontsize=6)
        plt.title(mods[i])

    plt.show()

def run_process():
    """ Process the data."""
    data_path = '../ModData/RML2016.10a_dict.dat'
    samples, mods = get_sample(data_path)
    plot_samples(samples, mods)

if __name__ == '__main__':
    run_process()