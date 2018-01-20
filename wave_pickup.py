#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを読み込み、NPZファイルとして保存する'
#

import wave
import argparse
import numpy as np

import matplotlib.pyplot as plt

from func import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('wav', nargs='+',
                        help='使用するWAVファイル')
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


def LPFilter(x, fc):
    N = len(x)
    # 高速フーリエ変換
    F = np.fft.fft(x) / (N / 2)
    # 周波数軸の値を計算
    freq = np.fft.fftfreq(N)
    # 直流成分の振幅を揃える
    F[0] = F[0] / 2
    # ローパス処理
    F[(freq > fc)] = 0
    F[(freq < 0)] = 0
    # 高速逆フーリエ変換
    return np.array(np.real(np.fft.ifft(F)) * (2 * N / 2), dtype=np.float32)


def waveCut(x, width, height):
    wave = []
    num = 0
    for i in range(len(x)):
        if(x.item(i) > height):
            print(i - num)
            num = i
            wave.append(x[i:i + width].copy())
            x[i:i + width] = 0

    return wave


def savePNG(save_folder, wave_list, fs=(0.5, 0.5),
            ylim=[0, 1], lw=0.4, dpi=200, infla_size=5):

    for i, elem in enumerate(wave_list):
        fig = plt.figure(figsize=fs)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(elem, linewidth=lw)
        ax.set_ylim(ylim[0], ylim[1])

        plt.savefig(getFilePath(save_folder, str(i).zfill(2), '.png'), dpi=dpi)
        plt.close()


def saveNPZ(save_folder, wave_list):
    np.savez(getFilePath(save_folder, 'wave', ''), wave=np.array(wave_list))


def show(waves):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for w in waves:
        ax.plot(w)

    plt.show()


def main(args):
    wave = [wav2arr(w) for w in args.wav]
    x = averageSampling(norm(wave[0]), 50)
    x[x > 0.20] = 0.2
    x = x * 5
    wave = waveCut(x, 300, 0.7)

    savePNG('./result/png', wave)
    saveNPZ('./result/', wave)
    show([wave[0]])


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
