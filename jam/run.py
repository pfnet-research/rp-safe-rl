import argparse
import json
import os
from datetime import datetime
from statistics import mean

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from chainerrl.wrappers import CastObservationToFloat32, ScaleReward

from estimator import RecoNet, SmallRecoNet, ThreatEstimator
from q_func import RPDQN, QFunction, RPQFunction
from train_agent_util import train_agent_with_evaluation
from util import Environment, Field

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run.py',
        description='run learning',
        add_help=True
    )

    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('-n', dest='n_agent', default=8, type=int)
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--load', dest='load', default='', type=str)
    parser.add_argument('--normal', dest='normal', action='store_true')
    parser.add_argument('--alllog', dest='all_log', action='store_true')
    parser.add_argument('--adamalpha', dest='adam_alpha',
                        default=1e-3, type=float)
    parser.add_argument('--cur', dest='cur', action='store_true')
    parser.add_argument('--lmd', dest='lmd', default=50, type=int)
    parser.add_argument('--scale', dest='scale', default=1.0, type=float)
    parser.add_argument('--starteps', dest='starteps', default=1.0, type=float)
    parser.add_argument('--step', dest='step', default=3 * 10 ** 6, type=int)

    parser.add_argument('--demo', dest='demo', action='store_true')

    parser.add_argument('--eval', dest='eval', type=str, default='')
    parser.add_argument('-t', dest='times', default=100, type=int)
    args = parser.parse_args()

    gpus = (0,) if args.gpu else ()

    gamma = 0.90
    step = args.step
    max_episode_len = 100

    chainerrl.misc.set_random_seed(args.seed, gpus)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join('results/jam', timestamp)
    os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    field = Field(height=3, width=3, exit_=0.8)
    env = Environment(n_others=args.n_agent, field=field,
                      render=args.render, result_dir=result_dir,
                      all_log=args.all_log, lmd=args.lmd,
                      max_step=step, cur=args.cur)

    n_actions = len(env.my.action_list)

    env = CastObservationToFloat32(env)
    env = ScaleReward(env, args.scale)

    if args.normal:
        q_func = QFunction(n_actions, n_agents=args.n_agent)

    else:
        wall_model = SmallRecoNet()
        wall_file = 'jam/wall-threat.model'
        obj_model = RecoNet()
        obj_file = 'jam/car-threat.model'
        estimator = ThreatEstimator(
            env.unwrapped, wall_model, wall_file, obj_model, obj_file, args.gpu)

        danger_limit = 0.01
        q_func = RPQFunction(n_actions,
                             estimator, danger_limit, n_agents=args.n_agent)

    optimizer = chainer.optimizers.Adam(eps=1e-2, alpha=args.adam_alpha)
    optimizer.setup(q_func)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        args.starteps, 0.05, step, random_action_func=lambda: np.random.randint(n_actions))

    replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(
        capacity=1e6)

    if args.normal:
        agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, gamma, explorer, clip_delta=False,
            replay_start_size=600, update_interval=1,
            target_update_interval=1e3)
    else:
        agent = RPDQN(
            q_func, optimizer, replay_buffer, gamma, explorer, clip_delta=False,
            replay_start_size=600, update_interval=1,
            target_update_interval=1e3)

    env.unwrapped.result_agent = agent

    if args.demo:
        if args.load:
            agent.load(args.load)

        env = Environment(n_others=args.n_agent, field=field,
                          render=True, result_dir=result_dir,
                          all_log=args.all_log, lmd=args.lmd,
                          max_step=step, cur=args.cur)
        env = CastObservationToFloat32(env)

        if not args.normal:
            agent.q_function.threat_predictor.env = env.unwrapped

        total_episode_reward = []

        for i in range(args.times):
            obs = env.reset()
            done = False
            total = 0
            st = 0

            while not done:
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                total += r
                st += 1
                num = '%03d' % st
                if st >= max_episode_len:
                    break
            print('Reward:', total)

    elif args.eval:
        def gen_dir_name(jobid):
            times = step // 10**5
            yield ''
            dirname = args.eval + '/'
            for i in range(times - 1):
                yield dirname + 'agent' + str(i + 1)
            yield dirname + str(int(step)) + '_finish'

        crash_ratio = []
        reward_list = []
        steps = np.arange(0, step + 1, 10**5)

        for agent_dir_name in gen_dir_name(args.eval):
            if agent_dir_name:
                agent.load(agent_dir_name)
            print('agent:', agent_dir_name)

            env = Environment(n_others=args.n_agent, field=field,
                              render=args.render, result_dir=result_dir,
                              all_log=args.all_log, lmd=args.lmd,
                              max_step=step, cur=args.cur)
            env = CastObservationToFloat32(env)

            if not args.normal:
                agent.q_function.threat_predictor.env = env.unwrapped

            total_episode_reward = []

            for i in range(args.times):
                obs = env.reset()
                done = False
                total = 0
                st = 0

                while not done:
                    action = agent.act(obs)
                    obs, r, done, _ = env.step(action)
                    total += r
                    st += 1
                    num = '%03d' % st
                    if st >= max_episode_len:
                        break

                if not env.unwrapped.crashed:
                    total_episode_reward.append(total)

            ave_reward = mean(total_episode_reward) if len(
                total_episode_reward) > 0 else np.nan
            ratio = env.unwrapped.crash_cnt / args.times

            print('result: crash_cnt ', ratio,
                  ' pure_reward ', ave_reward, end='\n\n')
            crash_ratio.append(ratio)
            reward_list.append(ave_reward)

        crash_ratio = np.array(crash_ratio)
        reward_list = np.array(reward_list)
        data = np.vstack((steps, crash_ratio))
        data2 = np.vstack((steps, reward_list))
        print(data)
        np.save(os.path.join(result_dir, 'crash.npy'), data)
        print(data2)
        np.save(os.path.join(result_dir, 'reward.npy'), data2)

    else:
        if args.load:
            agent.load(args.load)

        train_agent_with_evaluation(
            agent, env, steps=step, eval_n_steps=None, eval_n_episodes=5,
            train_max_episode_len=max_episode_len, eval_interval=1e4, outdir=result_dir,
            eval_env=env.env)
