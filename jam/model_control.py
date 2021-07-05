import argparse
import copy
from statistics import mean
from subprocess import call

import chainerrl
import matplotlib.pyplot as plt
import numpy as np
from mpc import calc

from util import Environment, Field

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='model_control.py',
        description='model predictive control',
        add_help=True
    )

    max_episode_len = 100

    parser.add_argument('-d', dest='depth', default=3, type=int)
    parser.add_argument('-t', dest='times', default=1, type=int)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    args = parser.parse_args()

    gpus = ()
    chainerrl.misc.set_random_seed(args.seed, gpus)

    field = Field(height=3, width=3, exit_=0.8)
    env = Environment(n_others=8, field=field, render=False,
                      all_log=True, lmd=50, max_step=max_episode_len)
    depth = args.depth

    log = []
    crash_cnt = 0
    for e in range(args.times):
        total = 0
        steps = 0

        obs = env.reset()
        for i in range(max_episode_len):
            other_pos = [agent.pos for agent in env.others]
            other_v = [agent.v for agent in env.others]
            other_front = [agent.front for agent in env.others]
            action_num = calc(depth, env.my.v, env.my.pos,
                              env.my.front, other_v, other_pos, other_front, field.wall)

            obs, r, done, _ = env.step(action_num)
            total += r
            steps += 1
            print('step', steps, ' / reward:', r, ' / total:', total)

            if done:
                print('DONE!!!!!')
                break

        print('\n*** result ***')
        print('step:', steps)
        print('reward:', total)
        print('---------------------------------------------')
        crash_cnt += env.crashed
        log.append(total)

    print('final_result:', mean(log))
    print('crash_ratio:', crash_cnt / args.times)
