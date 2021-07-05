import itertools
from subprocess import call

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import serializers
from chainer.backends import cuda


class SmallRecoNet(chainer.Chain):
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


class RecoNet(chainer.Chain):
    def __init__(self, n_hid1=500, n_hid2=500, n_hid3=500, n_out=15):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_hid1)
            self.l1 = L.Linear(None, n_hid2)
            self.l2 = L.Linear(None, n_hid3)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x, test=False):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        return self.l3(h)

    def predict(self, x, test=False):
        h = self.__call__(x, test=test)
        return F.sigmoid(h)


class ThreatEstimator:
    def __init__(self, env, wall_model, wall_file, obj_model, obj_file, gpu):
        self.env = env
        self.wall_model = wall_model
        self.gpu = gpu
        serializers.load_npz(wall_file, self.wall_model)
        if gpu:
            self.wall_model.to_gpu(0)

        self.obj_model = obj_model
        serializers.load_npz(obj_file, self.obj_model)
        if gpu:
            self.obj_model.to_gpu(0)

        acc_list = [-0.02, 0, 0.02]
        other_omega_list = [-0.15, -0.05, 0, 0.05, 0.15]
        omega_list = [-0.3, -0.1, 0, 0.1, 0.3]

        p0 = [0.2 * 0.2] * 5
        p1 = [0.6 * 0.2] * 5
        self.p_list = p0 + p1 + p0

        self.action_list = list(itertools.product(
            acc_list, omega_list))
        self.other_action_list = list(itertools.product(
            acc_list, other_omega_list))

    def threat(self, xp):
        r = self.env.wall_rpos
        rx, ry = r.real, r.imag
        mask = (-0.7 < rx) & (rx < 0.7) & (-0.5 < ry) & (ry < 0.5)
        r = r[mask]
        num = r.shape[0]

        if num is 0:
            wall_threat = xp.array([0] * len(self.action_list))
        else:
            v = xp.ones((num, 1)) * self.env.my.v
            data = xp.c_[r.real, r.imag, v].astype(np.float32)
            if self.gpu:
                data = cuda.cupy.asarray(data)
            wall_threat = self.wall_model.predict(data).array.sum(axis=0)
            if self.gpu:
                wall_threat = cuda.cupy.asnumpy(wall_threat)

        agents = self.env.near_agents()
        r = (xp.array([obj.pos for obj in agents]) -
             self.env.my.pos) / self.env.my.front
        rx, ry = r.real, r.imag
        mask = (-1.5 < rx) & (rx < 1.5) & (-1.5 < ry) & (ry < 1.5)
        r = r[mask]
        num = r.shape[0]

        if num is 0:
            obj_threat = xp.array([0] * len(self.action_list))
        else:
            myv = xp.ones((num, 1)) * self.env.my.v
            v = xp.array([obj.v for obj in agents]).reshape(-1, 1)[mask]
            th = xp.array([xp.angle(obj.front) for obj in agents]
                          ).reshape(-1, 1)[mask] - xp.angle(self.env.my.front)
            th = xp.angle(xp.exp(1.j * th))

            data = xp.c_[r.real, r.imag, th, v, myv].astype(np.float32)
            if self.gpu:
                data = cuda.cupy.asarray(data)
            obj_threat = self.obj_model.predict(data).array.sum(axis=0)
            if self.gpu:
                obj_threat = cuda.cupy.asnumpy(obj_threat)

        return wall_threat + obj_threat
