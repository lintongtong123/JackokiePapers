#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/15 15:19
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : data_show.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

import os
import scipy.io as sio
import pickle
import numpy as np

orig_path = 'E:\ModData\M.Q.Liu'
snrs = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
data_path = 'E:\ModData\mod_data.pkl'

def load_data(path):

    mods = os.listdir(path)
    data = dict()

    print('Analysing the data...')
    for mod in mods:
        mod_dir = path + '\\' + mod + '\\'
        files = sorted(os.listdir(mod_dir), key=len)
        for index in range(len(files)):
            file_path = mod_dir + files[index]
            temp_snr = list(sio.loadmat(file_path).values())[3]
            real = np.float32(np.real(temp_snr))
            imag = np.float32(np.imag(temp_snr))
            data[(mod, snrs[index])] = np.hstack((real, imag))

    pickle.dump(data, open(data_path, 'wb'))
    print('Data has been stored on the disk...')


if __name__ == '__main__':
    temp = load_data(orig_path)