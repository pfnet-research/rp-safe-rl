import argparse
import sys
from copy import copy, deepcopy
from itertools import product
from statistics import mean

import chainer
import chainerrl
import lasagne.nonlinearities as NL
import numpy as np
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

from dqn_util import DoubleDQN

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='mpc.py',
        description='PointGather model predictive control',
        add_help=True
    )

    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--depth', dest='depth', default=2, type=int)
    parser.add_argument('-t', dest='times', default=10, type=int)

    args = parser.parse_args()

    gpus = ()

    chainerrl.misc.set_random_seed(args.seed, gpus)

    env = PointGatherEnv(apple_reward=10, bomb_cost=1,
                         n_apples=2, activity_range=6)

    n_actions = 81
    max_episode_len = 15

    reward_list = []
    crash_list = []
    a = [i for i in range(n_actions)]
    for t in range(args.times):
        print('== episode', t, '==')
        env.reset()
        done = False
        total = 0
        st = 0

        # calc max reward move
        bomb_flag = False
        prev = -1
        while not done:
            print('* step', st)
            max_r = -1e6
            max_a = -1
            candi = []
            for x in product(a, repeat=args.depth):
                if x[0] != prev:
                    pass
                    #print('action', x[0])
                prev = x[0]

                e = deepcopy(env)
                e.objects = deepcopy(env.objects)

                d = e.inner_env.model.data
                ori = env.inner_env.model.data

                d.qpos = copy(ori.qpos)
                d.qvel = copy(ori.qvel)
                d.qacc = copy(ori.qacc)
                d.ctrl = copy(ori.ctrl)

                tot = 0
                for action in x:
                    ac = action_parser(action)
                    _, r, done_, _ = e.step(ac)
                    tot += r if r >= 0 else -200
                    if done_:
                        break

                if tot > max_r:
                    max_r = tot
                    max_a = x[0]
                    candi = [max_a]
                elif tot == max_r:
                    candi.append(x[0])

            candi = np.unique(candi)
            ac = np.random.choice(candi)
            print('max_r', max_r, 'selected', ac, 'candidate', candi)
            action = action_parser(ac)
            obs, r, done, _ = env.step(action)
            total += r
            if r < 0:
                bomb_flag = True
            st += 1
            if st >= max_episode_len:
                break

        print('reward', total)
        reward_list.append(total)
        crash_list.append(int(bomb_flag))

    print(reward_list)
    print('average', mean(reward_list))
    print('crash', mean(crash_list))
    print('total_bomb', env.total_bomb)
