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
random.seed(2018)


def get_sample(data_path):
    """Get the sample to show.
    Parameters:
        data_path: The data in the system.
    Returns:
        samples: The samples to show.
        mods: The samples' category.
    """
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')
    mode_snr = list(orig_data.keys())
    mods, snrs = [sorted(list(set(x[i] for x in mode_snr))) for i in [0, 1]]
    mods.remove('AM-SSB')
    mods.remove('WBFM')
    snr = 0
    samples = [orig_data[(mods[i], snr)][786] for i in range(len(mods))]
    return samples, mods

def freq_plot(samples, mods):
    """ Plot the samples.
    Parameters:
        samples: The samples to be plot.
        mods: The Modulation of samples.
    """

    plt.figure(num=9, figsize=(12, 10), dpi=600)

    comp_samples = list()
    for i in range(len(samples)):
        comp_samples.append([np.complex(samples[i][1][j], samples[i][0][j])
                                for j in range(128)])
    # freq_den = [20 * np.log10(np.fft.fft(samples[i][1]))
    #             for i in range(len(samples))]

    freq_den = [20 * np.log10(np.fft.fft(comp_samples[i]))
                for i in range(len(samples))]

    for i in range(len(mods)):
        plt.subplot(3, 3, i+1)
        plt.plot(freq_den[i])
        axes = plt.gca()
        # axes.set_ylim([-80, 0])
        # axes.set_xlim([0, 128])

        plt.xticks([])
        plt.yticks([])
        plt.title(mods[i])
    plt.savefig("E:/JackokiePapers/figures/chapter_3/fig_3_3.jpg")
    return

def time_plot(samples, mods):
    """ Plot the samples.
    Parameters:
        samples: The samples to be plot.
        mods: The Modulation of samples.
    """
    plt.figure(num=9, figsize=(10, 8), dpi=600)
    num = len(samples)
    for i in range(num):

        plt.subplot(3, 3, i+1)
        p1, = plt.plot(samples[i][0], 'blue')
        p2, = plt.plot(samples[i][1], 'green')
        plt.hlines(0, 0, 128)

        axes = plt.gca()
        # axes.set_ylim([-0.022, 0.022])
        # axes.set_xlim([0, 128])

        plt.xticks([])
        plt.yticks([])
        # plt.yticks([-0.02, 0.02])
        plt.legend([p1, p2], ['Real', 'Complex'], loc='upper right', fontsize=6)
        plt.title(mods[i])

    plt.savefig("E:/JackokiePapers/figures/chapter_3/fig_3_2.jpg", dpi=120)
    plt.clf()

def run_process():
    """ Process the data."""
    data_path = 'E:/ModData/RML2016.10a_dict.dat'
    samples, mods = get_sample(data_path)
    time_plot(samples, mods)
    freq_plot(samples, mods)

if __name__ == '__main__':
    run_process()