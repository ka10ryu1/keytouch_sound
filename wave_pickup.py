#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを読み込み、NPZファイルとして保存する'
#

import argparse
import matplotlib.pyplot as plt

from Tools.func import argsPrint
import Lib.wavefunc as W


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('wav', nargs='+',
                        help='使用するWAVファイル')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    parser.add_argument('-a', '--augmentation', type=int, default=100,
                        help='データ増幅倍率 (default: 100)')
    parser.add_argument('--save_png', action='store_true',
                        help='波形をPNG形式で保存（処理が重いので注意）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')
    return parser.parse_args()


def show(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    [ax.plot(x, w) for w in y]
    plt.show()


def main(args):
    wave = [W.wav2arr(w) for w in args.wav]
    x = W.amplifier(W.averageSampling(W.norm(wave[0]), 120))
    print(x.shape)
    show(range(len(x)), [x])
    wave = W.waveCut(x, 100, 0.6)
    wave = W.waveAugmentation(wave, args.augmentation)

    if args.save_png:
        W.savePNG(args.out_path, wave)

    W.saveNPZ(args.out_path, wave, args.train_per_all)
    show(range(len(wave[0])), wave[:3])


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
