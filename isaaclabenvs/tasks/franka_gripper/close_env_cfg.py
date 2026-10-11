from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .articulation_open_close_env_cfg import ArticulatedOpenCloseEnvCfg


@configclass
class CloseEnvCfg(ArticulatedOpenCloseEnvCfg):
    observation_space: int = 84

    target_joint_pos_norm: float = 0.0
    joint_start_pos_range_norm: tuple[float, float] = (0.999, 1.0)

    # done/success logic
    num_consecutive_success: int = 5
    # If > 0, terminate as FAILURE when the drawer is pulled back OPEN more than this (normalized)
    # from its closest-to-closed peak: (current - min_close_pos_norm) > regress_fail_norm. 0 = disabled.
    regress_fail_norm: float = 0.0

    # reward terms
    w_rew_close: float = 500.0
    w_rew_closed_once: float = 500.0
    w_rew_closed_hold: float = 20.0
    w_rew_dist: float = -0.5
    w_rew_contact: float = 1.0
    w_rew_success: float = 500.0

    # Per-link object-filtered contacts. After merge_fixed_joints, handle_link is absorbed into
    # movable_link (no separate rigid body / collider), so handle contacts now register against
    # movable_link directly.
    contact_left_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Object/movable_link.*",
        ],
        update_period=0.0,
        debug_vis=False,
    )
    contact_left_tip_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger_tip",
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Object/movable_link.*",
        ],
        update_period=0.0,
        debug_vis=False,
    )
    contact_right_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Object/movable_link.*",
        ],
        update_period=0.0,
        debug_vis=False,
    )
    contact_right_tip_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger_tip",
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Object/movable_link.*",
        ],
        update_period=0.0,
        debug_vis=False,
    )
