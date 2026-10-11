from isaaclab.utils import configclass

from .close_env_cfg import CloseEnvCfg


@configclass
class CloseDrawerEnvCfg(CloseEnvCfg):
    # CloseEnv (84) + drawer push (close) direction (3)
    observation_space: int = 87
    dataset: str = "arti_prismatic"
    # Reward weight for finger reaction force aligned with the close (push-in) direction.
    w_rew_force_align: float = 1.0
