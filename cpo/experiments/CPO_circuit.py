import argparse
import sys
from statistics import mean

import chainer
import chainerrl
import lasagne.nonlinearities as NL
import numpy as np
from cpo.algos.safe.cpo import CPO
from cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from cpo.experiments.circuit_util import Environment, default_circuit
from cpo.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from cpo.safety_constraints.circuit import CircuitSafetyConstraint
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from rllab.envs.env_spec import EnvSpec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CPO_circuit.py',
        description='run learning',
        add_help=True
    )

    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--seed', dest='seed', default=0, type=int)

    parser.add_argument('--alllog', dest='all_log', action='store_true')
    parser.add_argument('--lmd', dest='lmd', default=200, type=int)

    parser.add_argument('--step', dest='step', default=5 * 10 ** 6, type=int)
    parser.add_argument('--trpostep', dest='trpostep',
                        default=0.01, type=float)
    parser.add_argument('--batchsize', dest='batchsize',
                        default=3000, type=int)

    parser.add_argument('--ren', dest='ren', action='store_true')
    args = parser.parse_args()

    gpus = (0,) if args.gpu else ()

    chainerrl.misc.set_random_seed(args.seed, gpus)

    ###
    def run_task(*_):
        trpo_stepsize = args.trpostep
        trpo_subsample_factor = 0.2

        circuit = default_circuit()
        env = Environment(circuit=circuit,
                          random_init=True, file='crash_train.log', all_log=False,
                          lmd=args.lmd, render=False)

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

        safety_constraint = CircuitSafetyConstraint(
            max_value=1e-3, baseline=safety_baseline)

        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=1,
            batch_size=args.batchsize,
            max_path_length=200,
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
        exp_prefix='CPO-Circuit',
        seed=args.seed,
        python_command='python3',
        mode="local"
        # plot=True
    )

    ###
