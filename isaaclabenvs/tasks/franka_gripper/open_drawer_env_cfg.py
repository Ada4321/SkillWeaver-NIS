from isaaclab.utils import configclass

from .open_env_cfg import OpenEnvCfg


@configclass
class OpenDrawerEnvCfg(OpenEnvCfg):
    # OpenEnv (84) + drawer_goal_dir (3)
    observation_space: int = 87
    dataset: str = "arti_prismatic"
    w_rew_force_align: float = 1.0
    w_rew_wrist_dir_align: float = 1.0
    w_rew_finger_perp_align: float = 1.0
