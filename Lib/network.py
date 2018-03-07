#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'jpegcompのネットワーク部分'
#

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import Chain


class KBT(Chain):
    def __init__(self, n_in=1, n_unit=128, n_out=10,
                 layer=3, actfun=F.sigmoid, dropout=1.0):
        """
        [in] n_unit: 中間層のユニット数
        [in] n_out:  出力チャンネル
        [in] layer:  中間層の数
        [in] actfun: 活性化関数
        """

        super(KBT, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(None, n_unit)
            self.l2 = L.LSTM(None, n_unit)
            self.l3 = L.Linear(None, n_out)

        self.ratio = dropout
        print('[Network info]')
        print('  Unit:\t{0}\n  Out:\t{1}\n  Layer:\t{2}\n  Act Func:\t{3}'.format(
            n_unit, n_out, layer, actfun.__name__
        ))

    def __call__(self, x):
        self.reset_state()
        h = F.dropout(self.l1(x), self.ratio)
        h = F.dropout(self.l2(h), self.ratio/2)
        y = self.l3(h)
        return y

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()


class BPTTUpdater(training.StandardUpdater):
    # Custom updater for truncated BackProp Through Time (BPTT)

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
