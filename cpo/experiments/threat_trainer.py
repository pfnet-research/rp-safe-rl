import argparse
import random
import sys

import chainer
import chainer.links as L
import numpy as np
from chainer import iterators, serializers, training
from chainer.backends import cuda
from chainer.datasets import split_dataset_random, tuple_dataset
from chainer.training import extensions

from estimator import RecoNet
from sigmoid_cross_entropy_util import sigmoid_cross_entropy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='threat_trainer.py',
        description='train threat network',
        add_help=True
    )
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('-n', dest='n_epoch', type=int, default=20)
    parser.add_argument('--batch', dest='batch', type=int, default=512)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    args = parser.parse_args()

    xp = cuda.cupy if args.gpu else np
    device = 0 if args.gpu else -1
    train_val = xp.load('data/point_gather_threat.npy').astype(np.float32)
    print('data size :', len(train_val))

    train_val = tuple_dataset.TupleDataset(
        train_val[:, :9], train_val[:, 9:])
    n_train = int(len(train_val) * 0.8)
    train, val = split_dataset_random(train_val, n_train, seed=0)

    train_iter = iterators.SerialIterator(train, args.batch)
    valid_iter = iterators.SerialIterator(
        val, args.batch, repeat=False, shuffle=False)

    net = RecoNet(n_actions=81)
    model = L.Classifier(net, lossfun=sigmoid_cross_entropy)
    model.compute_accuracy = False

    if args.gpu:
        model.to_gpu(0)

    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)

    trainer = training.Trainer(updater, (args.n_epoch, 'epoch'), out='results')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(
        valid_iter, model, device=device), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'val/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()

    chainer.serializers.save_npz('cpo/experiments/threat.model', net)
