from isaaclab.utils import configclass

from .open_env_cfg import OpenEnvCfg


@configclass
class OpenDoorEnvCfg(OpenEnvCfg):
    # OpenEnv (84) + axis/radial/tangent dirs (9)
    observation_space: int = 93
    dataset: str = "arti_revolute"
    w_rew_force_align: float = 1.0
    w_rew_wrist_dir_align: float = 1.0
    w_rew_finger_perp_align: float = 1.0
