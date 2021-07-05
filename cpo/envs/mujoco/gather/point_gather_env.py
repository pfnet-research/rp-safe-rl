from cpo.envs.mujoco.gather.gather_env import GatherEnv
from cpo.envs.mujoco.point_env import PointEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
