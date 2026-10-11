from isaaclab.utils import configclass

from .close_env_cfg import CloseEnvCfg


@configclass
class CloseDoorEnvCfg(CloseEnvCfg):
    # CloseEnv (84) + axis/radial/close-tangent dirs (9)
    observation_space: int = 93
    dataset: str = "arti_revolute"
    # Probability of flipping yaw so the handle faces AWAY from y=0 instead of toward
    # (default 0 = always toward). Applied per-env on revolute joints only.
    yaw_away_prob: float = 0.0
    # Per-env Uniform(-yaw_jitter_rad, +yaw_jitter_rad) added to theta after the toward/away
    # choice (rad). Applied on revolute joints only.
    yaw_jitter_rad: float = 0.0
    # Reward weight for finger reaction torque aligned with the close-direction around the hinge.
    w_rew_force_align: float = 1.0
