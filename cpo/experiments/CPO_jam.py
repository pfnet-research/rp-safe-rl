import argparse
import sys
from statistics import mean

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import lasagne.nonlinearities as NL
import numpy as np
from chainerrl.wrappers import CastObservationToFloat32, ScaleReward
from cpo.algos.safe.cpo import CPO
from cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from cpo.experiments.jam_util import Environment, Field
from cpo.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from cpo.safety_constraints.jam import JamSafetyConstraint
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from rllab.envs.env_spec import EnvSpec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CPO_jam.py',
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
    parser.add_argument('--clip', dest='clip', action='store_true')
    parser.add_argument('--adamalpha', dest='adam_alpha',
                        default=1e-3, type=float)
    parser.add_argument('--cur', dest='cur', action='store_true')
    parser.add_argument('--lmd', dest='lmd', default=50, type=int)
    parser.add_argument('--scale', dest='scale', default=1.0, type=float)
    parser.add_argument('--starteps', dest='starteps', default=1.0, type=float)
    parser.add_argument('--step', dest='step', default=3 * 10 ** 6, type=int)
    parser.add_argument('--trpostep', dest='trpostep',
                        default=0.01, type=float)
    parser.add_argument('--batchsize', dest='batchsize',
                        default=3000, type=int)

    parser.add_argument('--demo', dest='demo', action='store_true')

    parser.add_argument('--eval', dest='eval', type=str, default='')
    parser.add_argument('-t', dest='times', default=100, type=int)
    args = parser.parse_args()

    gpus = (0,) if args.gpu else ()

    gamma = 0.90
    step = args.step
    max_episode_len = 100

    chainerrl.misc.set_random_seed(args.seed, gpus)

    ###
    def run_task(*_):
        trpo_stepsize = args.trpostep
        trpo_subsample_factor = 0.2

        field = Field(height=3, width=3, exit_=0.8)
        env = Environment(n_others=args.n_agent, field=field,
                          render=args.render, all_log=args.all_log, lmd=args.lmd,
                          max_step=step, cur=args.cur)

        policy = GaussianMLPPolicy(env.spec,
                                   hidden_sizes=(64, 32)
                                   )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                'hidden_sizes': (64, 32),
                'hidden_nonlinearity': NL.tanh,
                'learn_std': False,
                'step_size': trpo_stepsize,
                'optimizer': ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
            }
        )

        safety_baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                'hidden_sizes': (64, 32),
                'hidden_nonlinearity': NL.tanh,
                'learn_std': False,
                'step_size': trpo_stepsize,
                'optimizer': ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
            },
            target_key='safety_returns',
        )

        safety_constraint = JamSafetyConstraint(
            max_value=0.01, baseline=safety_baseline)

        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=1,
            batch_size=args.batchsize,
            max_path_length=max_episode_len,
            n_itr=int(args.step / args.batchsize),
            gae_lambda=0.95,
            discount=0.995,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor': trpo_subsample_factor},
            # plot=True,
        )

        algo.train()

    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix='CPO-Jam',
        seed=args.seed,
        python_command='python3',
        mode="local"
        # plot=True
    )

    ###
