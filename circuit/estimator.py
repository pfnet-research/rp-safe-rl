import itertools
from subprocess import call

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import serializers
from chainer.backends import cuda

from util import invLIDAR


class RecoNet(chainer.Chain):
    def __init__(self, n_hid1=100, n_hid2=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_hid1)
            self.l1 = L.Linear(None, n_hid2)
            self.l2 = L.Linear(None, 15)

    def __call__(self, x, test=False):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)

    def predict(self, x, test=False):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return F.sigmoid(self.l2(h))


class ThreatEstimator:
    def __init__(self, model, filename, gpu):
        self.model = model
        serializers.load_npz(filename, self.model)
        self.gpu = gpu
        if gpu:
            self.model.to_gpu(0)
        acc_list = [-0.02, 0, 0.02]
        omega_list = [-0.15, -0.05, 0, 0.05, 0.15]
        self.action_list = list(itertools.product(
            acc_list, omega_list))

    def threat(self, obs):
        xp = cuda.get_array_module(obs)
        r = invLIDAR(obs[:, :360][0], xp)
        v_val = obs[:, 363][0]

        rx, ry = r.real, r.imag
        mask = (-0.5 < rx) & (rx < 0.7) & (-0.3 < ry) & (ry < 0.3)
        r = r[mask]
        num = len(r)

        v = xp.ones((num, 1)) * v_val
        data = xp.c_[r.real, r.imag, v].astype(np.float32)
        if self.gpu:
            data = cuda.cupy.asarray(data)

        result = self.model.predict(data).array.sum(axis=0)
        if self.gpu:
            result = cuda.cupy.asnumpy(result)
        return result
