#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを読み込み、NPZファイルとして保存する'
#

import argparse
import numpy as np
import matplotlib.pyplot as plt

from Tools.func import argsPrint
import Lib.wavefunc as W


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('wav', nargs='+',
                        help='使用するWAVファイル')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.95,
                        help='画像数に対する学習用画像の割合（default: 0.95）')
    parser.add_argument('-a', '--augmentation', type=int, default=1,
                        help='データ増幅倍率 (default: 1)')
    parser.add_argument('--save_png', action='store_true',
                        help='波形をPNG形式で保存（処理が重いので注意）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    return parser.parse_args()


def show(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    [ax.plot(i, j) for i, j in zip(x, y)]
    plt.show()


def main(args):
    wave = [W.wav2arr(w) for w in args.wav]
    out_x = []
    out_y = []
    old_x = 0
    for w in wave:
        x = W.amplifier(W.averageSampling(W.norm(w), 120))
        x = x[int(len(x)*0.01):int(len(x)*0.99)]
        x, y = W.waveCut(x, 105, 0.6, offset=old_x)
        old_x = x[-1][-1]+1000
        out_x.extend(x)
        out_y.extend(y)
        print('total wave num:', len(out_x), len(out_y))

    wave = W.waveAugmentation(out_x, out_y, args.augmentation)

    if args.save_png:
        W.savePNG(args.out_path, wave)

    if len(wave) % 9 == 0:
        W.saveNPZ(args.out_path, wave, args.train_per_all)

    show(out_x, out_y)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
