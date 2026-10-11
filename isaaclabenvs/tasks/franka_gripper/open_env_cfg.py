from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .articulation_open_close_env_cfg import ArticulatedOpenCloseEnvCfg


@configclass
class OpenEnvCfg(ArticulatedOpenCloseEnvCfg):
    observation_space: int = 84

    target_joint_pos_norm: float = 0.95
    joint_start_pos_range_norm: tuple[float, float] = (0.0, 0.25)

    # done/success logic
    num_consecutive_success: int = 5
    # If > 0, terminate as FAILURE when the drawer falls back more than this (normalized) from its
    # peak opening: (max_open_pos_norm - current) > regress_fail_norm. 0 = disabled. Opt in via CLI.
    regress_fail_norm: float = 0.0

    # reward terms
    w_rew_open: float = 500.0
    # If true, only give/open-progress reward when pinch is active (hard gating).
    # If false, use soft scaling by (1 + rew_open_pinch_bonus * pinch).
    rew_open_require_pinch: bool = False
    rew_open_pinch_bonus: float = 1.0
    # If true, rew_open is also gated by direction constraints.
    rew_open_require_direction: bool = True
    # Max angular error (deg) for wrist direction target.
    rew_open_direction_wrist_max_angle_deg: float = 30.0
    # Max angular deviation (deg) from perpendicular between finger line and handle axis.
    rew_open_direction_finger_perp_max_angle_deg: float = 30.0
    w_rew_jaw_open: float = -0.2
    jaw_close_thres: float = 0.002
    # Encourage stable two-sided pinch near handle.
    w_rew_pinch: float = 1.0
    # Extra contact reward when pinch is active.
    w_rew_contact_pinch_bonus: float = 1.0
    # For thin/elongated handles grasped at the fingertips, restrict the validated-contact count
    # (rew_contact / bonus) and the pinch test to the two fingertip keypoints (cols 2,3) instead
    # of all four finger/tip keypoints. Default False keeps the original 4-keypoint behavior.
    # NOTE: rew_dist still uses all four keypoints.
    contact_use_tips_only: bool = False
    w_rew_opened_once: float = 500.0
    w_rew_opened_hold: float = 20.0
    w_rew_dist: float = -0.5
    w_rew_contact: float = 1.0
    w_rew_success: float = 500.0
    # If true, success requires both joint target reached and direction constraints.
    success_require_direction: bool = True
    # Max angular error (deg) for wrist direction target in success check.
    success_direction_wrist_max_angle_deg: float = 30.0
    # Max angular deviation (deg) from perpendicular in success check.
    success_direction_finger_perp_max_angle_deg: float = 30.0
    # Soft distance gate for pose-alignment rewards: gate = exp(-alpha * min_handle_dist).
    rew_pose_align_dist_alpha: float = 20.0

    # Per-link handle-filtered contacts (1 source link per sensor to satisfy PhysX tensor pairing).
    # NOTE: Generic `contact` is inherited from BaseEnvCfg; open-specific contact reward uses the 4 sensors below.
    contact_left_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        filter_prim_paths_expr=["/World/envs/env_.*/Object/movable_link.*"],
        update_period=0.0,
        debug_vis=False,
    )
    contact_left_tip_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger_tip",
        filter_prim_paths_expr=["/World/envs/env_.*/Object/movable_link.*"],
        update_period=0.0,
        debug_vis=False,
    )
    contact_right_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        filter_prim_paths_expr=["/World/envs/env_.*/Object/movable_link.*"],
        update_period=0.0,
        debug_vis=False,
    )
    contact_right_tip_handle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger_tip",
        filter_prim_paths_expr=["/World/envs/env_.*/Object/movable_link.*"],
        update_period=0.0,
        debug_vis=False,
    )
