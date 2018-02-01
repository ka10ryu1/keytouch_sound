#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'WAVファイルを読み込み、ランダムに推論実行テストを実行する'
#
import os
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer.cuda import to_cpu

from Lib.network import KBT

from Tools.func import argsPrint, fileFuncLine
import Lib.wavefunc as W


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('wav',
                        help='使用するWAVファイル')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    parser.add_argument('-af', '--actfun', default='sigmoid',
                        help='活性化関数 (default: sigmoid, other: relu, elu, c_relu, l_relu, h_sigmoid, tanh, s_plus)')
    parser.add_argument('-u', '--unit', type=int, default=64,
                        help='ネットワークのユニット数 (default: 64)')
    parser.add_argument('-ln', '--layer_num', type=int, default=3,
                        help='ネットワーク層の数 (default: 3)')
    parser.add_argument('-n', '--test_num', type=int, default=30,
                        help='試行回数ネットワークのユニット数 (default: 30)')
    return parser.parse_args()


def npfloat32(x):
    return np.asarray(x, dtype=np.float32)


def npint32(x):
    return np.asarray(x, dtype=np.int32)


def checkModelType(path):
    """
    入力されたパスが.modelか.snapshotかそれ以外か判定し、
    load_npzのpathを設定する
    [in]  path:      入力されたパス
    [out] load_path: load_npzのpath
    """

    # 拡張子を正とする
    name, ext = os.path.splitext(os.path.basename(path))
    load_path = ''
    if(ext == '.model'):
        print('model read')
    elif(ext == '.snapshot'):
        print('snapshot read')
        load_path = 'updater/model:main/'
    else:
        print('model read error')
        print(W.fileFuncLine())
        exit()

    return load_path


def main(args):
    wave = W.wav2arr(args.wav)
    x = W.amplifier(W.averageSampling(W.norm(wave), 120))
    x = x[int(len(x)*0.01):int(len(x)*0.99)]
    x, y = W.waveCut(x, 105, 0.6)
    x = npfloat32(x)
    key_num = 9
    key_list = list(range(1, key_num + 1)) * int(len(wave) // key_num)
    key = npint32(key_list)
    print(x.shape, key.shape)

    # 活性化関数を取得する
    actfun = W.getActfun(args.actfun)
    # モデルを決定する
    model = L.Classifier(
        KBT(n_in=1, n_unit=args.unit, layer=args.layer_num, actfun=actfun),
    )
    # load_npzのpath情報を取得する
    load_path = checkModelType(args.model)
    # 学習済みモデルの読み込み
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(fileFuncLine())
        exit()

    cnt = 0
    for i in range(args.test_num):
        label = np.random.randint(0, len(x))
        xx = np.array(x[label]).reshape(1, -1)
        y = model.predictor(xx)
        y = to_cpu(y.array)
        rslt = 'X'
        if(key[label] == y.argmax(axis=1)[0]):
            rslt = 'O'
            cnt += 1

        print('ans={0}, predict={1}, result={2}'.format(
            key[label], y.argmax(axis=1)[0], rslt)
        )

    print('Total Result:{0:5.2f}%'.format(cnt / args.test_num * 100))


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
