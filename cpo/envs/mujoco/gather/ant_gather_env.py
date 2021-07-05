from cpo.envs.mujoco.gather.gather_env import GatherEnv
from cpo.envs.mujoco.ant_env import AntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
