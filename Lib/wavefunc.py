#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを処理する'
#

import os
import sys
import wave
import numpy as np

import chainer.functions as F
import chainer.optimizers as O
import matplotlib.pyplot as plt

[sys.path.append(d) for d in ['../Tools/'] if os.path.isdir(d)]
from Tools.func import getFilePath, fileFuncLine


def wav2arr(file_name):
    wave_file = wave.open(file_name, 'r')  # Open
    print('ch:', wave_file.getnchannels())  # モノラルorステレオ
    print('frame rate:', wave_file.getframerate())  # サンプリング周波数
    print('nframes:', wave_file.getnframes())  # フレームの総数
    x = wave_file.readframes(wave_file.getnframes())  # frameの読み込み
    x = np.frombuffer(x, dtype=np.int16)  # numpy.arrayに変換
    return x


def norm(x, val=2**15):
    return np.abs(x / 2**15)


def averageSampling(x, ave):
    out_size = x.shape[0] // ave
    in_size = out_size * ave
    print('sampling: {0} -> {1}'.format(in_size, out_size))
    x = np.array(x[:in_size]).reshape(-1, ave)
    b = np.ones(ave) / ave
    return np.dot(x, b)


def amplifier(x, thresh=0.2):
    x[x > thresh] = thresh
    return x / thresh


def waveCut(w, width, height):
    x = []
    y = []
    for i in range(len(w)):
        if(w.item(i) > height):
            x.append(list(range(i, i+width)))
            y.append(w[i:i + width].copy())
            w[i:i + width] = 0

    return x, y


def savePNG(save_folder, wave_list, fs=(0.5, 0.5),
            ylim=[0, 1], lw=0.4, dpi=200, infla_size=5):

    print('save png...')
    for i, elem in enumerate(wave_list):
        fig = plt.figure(figsize=fs)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(elem, linewidth=lw)
        ax.set_ylim(ylim[0], ylim[1])
        f = os.path.join(save_folder, 'png')
        file_path = getFilePath(f, str(i).zfill(2), '.png')
        plt.savefig(file_path, dpi=dpi)
        plt.close()


def saveNPZ(save_folder, wave_list, train_per_all):
    shuffle = np.random.permutation(range(len(wave_list)))
    train_size = int(len(wave_list) * train_per_all)
    wave = np.array(wave_list)

    key_num = 9
    key_list = list(range(1, key_num + 1)) * int(len(wave_list) // key_num)
    key = np.array(key_list)
    print(wave.shape, key.shape)
    train_x = wave[shuffle[:train_size]]
    train_y = key[shuffle[:train_size]]
    test_x = wave[shuffle[train_size:]]
    test_y = key[shuffle[train_size:]]
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    print('save npz...')
    np.savez(getFilePath(save_folder, 'train', ''),
             x=np.array(train_x),
             y=np.array(train_y),
             )
    np.savez(getFilePath(save_folder, 'test', ''),
             x=np.array(test_x),
             y=np.array(test_y),
             )


def waveAugmentation(wave, num):
    noized = []
    wave_len = len(wave[1][1])
    for i in range(num):
        noized.extend([(i, j + np.random.uniform(-0.1, 0.1, wave_len)) for i, j in wave])

    return noized


def getActfun(actfun_str):
    if(actfun_str.lower() == 'relu'):
        actfun = F.relu
    elif(actfun_str.lower() == 'elu'):
        actfun = F.elu
    elif(actfun_str.lower() == 'c_relu'):
        actfun = F.clipped_relu
    elif(actfun_str.lower() == 'l_relu'):
        actfun = F.leaky_relu
    elif(actfun_str.lower() == 'sigmoid'):
        actfun = F.sigmoid
    elif(actfun_str.lower() == 'h_sigmoid'):
        actfun = F.hard_sigmoid
    elif(actfun_str.lower() == 'tanh'):
        actfun = F.hard_sigmoid
    elif(actfun_str.lower() == 's_plus'):
        actfun = F.softplus
    else:
        actfun = F.relu
        print('\n[Warning] {0}\n\t{1}->{2}\n'.format(
            fileFuncLine(), actfun_str, actfun.__name__)
        )

    print('Activation func:', actfun.__name__)
    return actfun


def getOptimizer(opt_str):
    if(opt_str.lower() == 'adam'):
        opt = O.Adam()
    elif(opt_str.lower() == 'ada_d'):
        opt = O.AdaDelta()
    elif(opt_str.lower() == 'ada_g'):
        opt = O.AdaGrad()
    elif(opt_str.lower() == 'm_sgd'):
        opt = O.MomentumSGD()
    elif(opt_str.lower() == 'n_ag'):
        opt = O.NesterovAG()
    elif(opt_str.lower() == 'rmsp'):
        opt = O.RMSprop()
    elif(opt_str.lower() == 'rmsp_g'):
        opt = O.RMSpropGraves()
    elif(opt_str.lower() == 'sgd'):
        opt = O.SGD()
    elif(opt_str.lower() == 'smorms'):
        opt = O.SMORMS3()
    else:
        opt = O.Adam()
        print('\n[Warning] {0}\n\t{1}->{2}\n'.format(
            fileFuncLine(), opt_str, opt.__doc__.split('.')[0])
        )

    print('Optimizer:', opt.__doc__.split('.')[0])
    return opt
