#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを読み込み、NPZファイルとして保存する'
#

import wave
import argparse
import numpy as np

import matplotlib.pyplot as plt

from myfunc import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('wav', nargs='+',
                        help='使用するWAVファイル')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    return parser.parse_args()


def wav2arr(file_name):
    wave_file = wave.open(file_name, "r")  # Open
    print('ch:', wave_file.getnchannels())  # モノラルorステレオ
    print('frame rate:', wave_file.getframerate())  # サンプリング周波数
    print('nframes:', wave_file.getnframes())  # フレームの総数
    x = wave_file.readframes(wave_file.getnframes())  # frameの読み込み
    x = np.frombuffer(x, dtype="int16")  # numpy.arrayに変換
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


def waveCut(x, width, height):
    wave = []
    for i in range(len(x)):
        if(x.item(i) > height):
            wave.append(x[i:i + width].copy())
            x[i:i + width] = 0

    return wave


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
        plt.savefig(getFilePath(save_folder, str(i).zfill(2), '.png'), dpi=dpi)
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
    for i in range(num):
        wave_len = wave[0].shape[0]
        noized.extend([w + np.random.uniform(-0.1, 0.1, wave_len) for w in wave])

    return noized


def show(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    [ax.plot(x, w) for w in y]
    plt.show()


def main(args):
    wave = [wav2arr(w) for w in args.wav]
    x = amplifier(averageSampling(norm(wave[0]), 120))
    wave = waveCut(x, 100, 0.6)
    wave = waveAugmentation(wave, 100)
    #savePNG('./result/png', wave)
    saveNPZ('./result/', wave, args.train_per_all)
    show(range(len(wave[0])), wave[:3])


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
