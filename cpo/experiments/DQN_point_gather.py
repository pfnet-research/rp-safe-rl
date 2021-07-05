import argparse
import json
import os
import sys
from datetime import datetime
from statistics import mean

import chainer
import chainerrl
import lasagne.nonlinearities as NL
import numpy as np
from cpo.algos.safe.cpo import CPO
from cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from cpo.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from cpo.safety_constraints.gather import GatherSafetyConstraint
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from dqn_util import DoubleDQN
from estimator import RecoNet, ThreatEstimator
from q_func import RPDQN, QFunction, RPQFunction

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DQN_point_gather.py',
        description='run learning',
        add_help=True
    )

    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--normal', dest='normal', action='store_true')

    parser.add_argument('--limit', dest='limit', default=0.1, type=float)
    parser.add_argument('--adameps', dest='adam_eps', default=1e-2, type=float)
    parser.add_argument('--adamalpha', dest='adam_alpha',
                        default=1e-3, type=float)
    parser.add_argument('--gamma', dest='gamma', default=0.90, type=float)
    parser.add_argument('--lmd', dest='lmd', default=1, type=int)
    parser.add_argument('--clip', dest='clip', action='store_true')
    parser.add_argument('--step', dest='step', default=1e6, type=str)

    parser.add_argument('--load', dest='load', default='', type=str)
    parser.add_argument('--eval', dest='eval', type=str, default='')
    parser.add_argument('-t', dest='times', default=100, type=int)

    args = parser.parse_args()

    gpus = (0,) if args.gpu else ()

    chainerrl.misc.set_random_seed(args.seed, gpus)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join('results/gather', timestamp)
    os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    env = PointGatherEnv(apple_reward=10, bomb_cost=args.lmd,
                         n_apples=2, activity_range=6)

    n_actions = 81
    danger_limit = args.limit
    step = int(args.step)
    max_episode_len = 15

    if args.normal:
        q_func = QFunction(n_actions)
    else:
        reconet = RecoNet(n_actions=n_actions)
        estimator = ThreatEstimator(
            reconet, 'cpo/experiments/threat.model', args.gpu)
        q_func = RPQFunction(n_actions, estimator, danger_limit)

    optimizer = chainer.optimizers.Adam(
        eps=args.adam_eps, alpha=args.adam_alpha)
    optimizer.setup(q_func)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        1.0, 0.05, step, random_action_func=lambda: np.random.randint(n_actions))

    replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(1e6)

    if args.normal:
        agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, args.gamma, explorer, clip_delta=args.clip,
            replay_start_size=600, update_interval=1,
            target_update_interval=1e3, phi=lambda x: x.astype(np.float32))
    else:
        agent = RPDQN(
            q_func, optimizer, replay_buffer, args.gamma, explorer, clip_delta=args.clip,
            replay_start_size=600, update_interval=1,
            target_update_interval=1e3, phi=lambda x: x.astype(np.float32))

    env.result_agent = agent

    if args.eval:
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

            env = PointGatherEnv(
                apple_reward=10, bomb_cost=1, n_apples=2, activity_range=6, log=True)

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

                if not env.crashed:
                    total_episode_reward.append(total)

            ave_reward = mean(total_episode_reward) if len(
                total_episode_reward) > 0 else np.nan
            ratio = env.total_bomb / args.times

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

        chainerrl.experiments.train_agent_with_evaluation(
            agent, env, steps=step, eval_n_steps=None, eval_n_episodes=5,
            eval_env=PointGatherEnv(
                apple_reward=10, bomb_cost=1, n_apples=2, activity_range=6, log=False),
            train_max_episode_len=max_episode_len, eval_interval=2e3, outdir=result_dir)
