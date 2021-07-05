import argparse
import os
import random
import sys

import chainer
import chainerrl
import lasagne.nonlinearities as NL
import numpy as np
from chainerrl.wrappers import CastObservationToFloat32, ScaleReward
from cpo.algos.safe.cpo import CPO
from cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from cpo.envs.mujoco.gather.gather_env import action_parser
from cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from cpo.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from cpo.safety_constraints.gather import GatherSafetyConstraint
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from q_func import RPDQN, QFunction, RPQFunction

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sample_point_gather.py',
        description='sample threat vallue for approximation',
        add_help=True
    )

    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--beta', dest='beta', default=0.999, type=float)
    parser.add_argument('-d', dest='depth', default=5, type=int)
    parser.add_argument('-t', dest='times', default=1e6, type=float)

    args = parser.parse_args()

    gpus = ()

    chainerrl.misc.set_random_seed(args.seed, gpus)

    env = PointGatherEnv(apple_reward=10, bomb_cost=1,
                         n_apples=0, n_bombs=1, activity_range=6, log=False, sample=True)

    n_actions = 81

    def act():
        w = np.random.randint(9)
        if random.uniform(0, 1) < 0.1:
            a = np.random.randint(4)
        else:
            a = np.random.randint(5) + 4
        return a * 9 + w

    buf = []
    for t in range(int(args.times)):
        if t and t % 1000 is 0:
            print(t // 1000, 'k samples')
        cost = 0

        obs = env.reset()

        bomb = obs[-10:]
        ma = bomb.max()

        body = obs[:6]
        action = act()
        idx = bomb.argmax() if ma > 0 else 0
        bit = int(ma <= 0)
        data = np.append(body, (idx, ma, bit, action))

        action = action_parser(action)
        obs, r, done, _ = env.step(action)
        cost += r * args.beta

        for step in range(args.depth - 1):
            if done:
                break
            ma = bomb.max()
            action = act()
            action = action_parser(action)
            obs, r, done, _ = env.step(action)
            cost += r * (args.beta ** (step + 2))

        buf.append(np.append(data, -cost))

    os.makedirs('data', exist_ok=True)
    np.save('data/point_gather_threat.npy', buf)
