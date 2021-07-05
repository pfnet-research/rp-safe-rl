import argparse
import sys

import chainerrl
import lasagne.nonlinearities as NL
from cpo.algos.safe.cpo import CPO
from cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from cpo.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from cpo.safety_constraints.gather import GatherSafetyConstraint
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

sys.path.append(".")
ec2_mode = False

parser = argparse.ArgumentParser(
    prog='CPO_point_gather.py',
    description='run learning',
    add_help=True
)

parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--seed', dest='seed', default=0, type=int)

parser.add_argument('--limit', dest='limit', default=0.1, type=float)

args = parser.parse_args()

gpus = (0,) if args.gpu else ()

chainerrl.misc.set_random_seed(args.seed, gpus)


def run_task(*_):
    trpo_stepsize = 0.01
    trpo_subsample_factor = 0.2

    env = PointGatherEnv(apple_reward=10, bomb_cost=1,
                         n_apples=2, activity_range=6)

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

    safety_constraint = GatherSafetyConstraint(
        max_value=args.limit, baseline=safety_baseline)

    algo = CPO(
        env=env,
        policy=policy,
        baseline=baseline,
        safety_constraint=safety_constraint,
        safety_gae_lambda=1,
        batch_size=50000,
        max_path_length=15,
        n_itr=100,
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
    exp_prefix='CPO-PointGather',
    seed=args.seed,
    python_command='python3',
    mode="ec2" if ec2_mode else "local"
    # plot=True
)
