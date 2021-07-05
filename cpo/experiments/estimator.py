import itertools
from subprocess import call

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import serializers
from chainer.backends import cuda


class RecoNet(chainer.Chain):
    def __init__(self, n_hid=100, n_actions=81):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_hid)
            self.l1 = L.Linear(None, n_hid)
            self.l2 = L.Linear(None, n_actions)

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

    def threat(self, obs):
        xp = cuda.get_array_module(obs)

        obs = obs[0]
        body = obs[:6]
        bomb = obs[-10:]

        data = []

        for i, b in enumerate(bomb):
            if b > 0:
                data.append(np.append(body, (i, b, 0)))

        for i in range(8 - len(data)):
            data.append(np.append(body, (0, 0, 1)))

        data = xp.array(data, dtype=np.float32)

        if self.gpu:
            data = cuda.cupy.asarray(data)

        result = self.model.predict(data).array.sum(axis=0)

        if self.gpu:
            result = cuda.cupy.asnumpy(result)

        np.set_printoptions(2)
        return result
