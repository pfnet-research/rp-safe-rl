import argparse
import copy
from statistics import mean
from subprocess import call

import chainerrl
import matplotlib.pyplot as plt
import numpy as np
from mpc import calc

from util import Environment, default_circuit, invLIDAR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='model_control.py',
        description='model predictive control',
        add_help=True
    )
    parser.add_argument('-t', dest='timestep', default=4, type=int)
    parser.add_argument('-n', dest='n_episode', default=1, type=int)
    parser.add_argument('--crash', dest='crash', action='store_true')
    # parser.add_argument('--label', dest='label',
    #                    default='', type=str, required=True)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    args = parser.parse_args()

    gpus = ()
    chainerrl.misc.set_random_seed(args.seed, gpus)

    env = Environment(circuit=default_circuit(), random_init=args.crash)

    depth = args.timestep

    max_episode_len = 200

    total_reward = 0
    crash_cnt = 0
    reward = []
    crash = []

    for e in range(args.n_episode):
        print('\n----------------\nEPISODE', e)

        obs = env.reset()
        total = 0
        steps = 0

        traj = []

        for i in range(max_episode_len):
            r = invLIDAR(obs[:360], np)

            action_num = calc(depth, env.agent.v, env.agent.pos,
                              env.agent.front, env.circuit.wall, env.circuit.cpos)

            obs, r, done, _ = env.step(action_num)
            traj.append(env.agent.pos)

            total += r
            steps += 1
            print('step', steps, ' / reward:', r, ' / total:', total, '\n')

            # env.render()
            num = '%03d' % steps
            # plt.savefig('results/'+num+'.png')

            if done:
                print('DONE!!!!!')
                break

        print('\n*** result ***')
        print('step:', steps)
        print('reward:', total)

        # if e is 0:
        #    traj = np.array(traj)
        #    np.save('results/traj-'+args.label+'.npy', traj)

        crash.append(int(env.crashed))
        reward.append(total)

    print('crash', crash)
    print('reward', reward)
    #np.save('results/crash-'+args.label+'.npy', crash)
    #np.save('results/reward-'+args.label+'.npy', reward)

    print('AVERAGE REWARD ->', mean(reward))
    print('CRASH RATIO ->', mean(crash))
    print('random_init ->', args.crash)
