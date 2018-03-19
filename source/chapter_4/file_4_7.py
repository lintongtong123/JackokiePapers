#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:41
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_4_7.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

import os
import pickle

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

log_dir = './log/'
orig_file_path = 'E:/ModData/RML2016.10a_dict.dat'
[height, width] = [2, 128]
num_classes = 7
seed = 'jackokie'
train_ratio = 0.9


def load_data(data_path):
    """ Load the original data for training...
    Parameters:
        data_path: The original data path.
    Returns:
        train_data: Training data structured.
    """
    # load the original data.
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')

    # Get the set of snr & modulations
    mode_snr = list(orig_data.keys())
    mods, snrs = [sorted(list(set(x[i] for x in mode_snr))) for i in [0, 1]]
    mods.remove('AM-DSB')
    mods.remove('WBFM')
    mods.remove('8PSK')
    mods.remove('QAM16')

    # Build the train set.
    samples = []
    labels = []
    samples_snr = []
    mod2cate = dict()
    cate2mod = dict()
    for cate in range(len(mods)):
        cate2mod[cate] = mods[cate]
        mod2cate[mods[cate]] = cate

    for snr in snrs:
        for mod in mods:
            data_complex = [[np.complex(real[i], image[i])
                             for i in range(len(real))]
                            for real, image in orig_data[(mod, snr)]]
            samples.extend(data_complex)
            labels.extend(1000 * [mod2cate[mod]])
            samples_snr.extend(1000 * [snr])

    samples_snr = np.array(samples_snr)
    labels = np.array(labels)
    return samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr


def cum(samples):
    # 二阶矩
    m_20 = np.mean(np.square(samples))
    m_21 = np.mean(samples * np.conj(samples))

    # 四阶矩
    m_40 = np.mean(samples * samples * samples * samples)
    m_41 = np.mean(samples * samples * samples * np.conj(samples))
    m_42 = np.mean(samples * samples * np.conj(samples) * np.conj(samples))

    # 六阶矩
    m_60 = np.mean(np.power(samples, 6))
    m_63 = np.mean(np.power(samples, 3) * np.power(np.conj(samples), 3))

    # 二阶累积量
    c_20 = np.mean(np.square(samples))
    c_21 = np.mean(samples * np.conj(samples))

    # 四阶累积量
    c_40 = m_40 - 3 * m_20
    c_41 = m_41 - 3 * m_20 * m_21
    c_42 = m_42 - m_20 * m_20 - m_21 * m_21

    # 六阶累积量
    c_60 = m_60 - 15 * m_40 * m_20 + 30 * np.power(m_20, 3)
    c_63 = m_63 - 9 * c_42 * c_21 - 6 * np.power(c_21, 3)

    return c_20, c_21, c_40, c_41, c_42, c_60, c_63


def main():
    # Load the train data and test data.
    samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr = \
        load_data(orig_file_path)

    c_20, c_21, c_40, c_41, c_42, c_60, c_63 = cum(samples)
    

    print()


if __name__ == '__main__':
    main()
