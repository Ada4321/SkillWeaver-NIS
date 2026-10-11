from isaaclab.utils import configclass

from .base_env_cfg import BaseEnvCfg


@configclass
class PlaceEnvCfg(BaseEnvCfg):
    # env
    observation_space: int = 94
    state_space: int = 0

    # mode
    threshold_decay: bool = False
    require_orientation: bool = True
    # If enabled, orientation reward/success only depends on tilt (goal/object up-vector angle),
    # and ignores yaw difference around the up axis.
    require_tilt_only: bool = False
    # Tilt-only orientation threshold in degrees (used when require_tilt_only=True).
    tilt_rot_tol_deg: float = 17.2
    # When True together with require_tilt_only=True, the tilt relaxation only applies to
    # upright-mode assets; non-upright (lying / mixed) envs fall back to the standard
    # require_orientation logic (full orientation if require_orientation=True, otherwise
    # no orientation requirement).
    tilt_only_for_upright_only: bool = False
    reach_once_pos_only: bool = False
    post_reset_forced_steps: int = 0

    # reward settings
    num_consecutive_success = 5

    # dataset
    dataset: str = "rigid"
    dataset_num_grasp: int = 400
    # Grasp sampling mode:
    # - mixed: sample from all grasp ids
    # - general: sample from [0, grasp_mode_split_1)
    # - upright: sample from [grasp_mode_split_1, grasp_mode_split_2)
    # - convex: sample from [grasp_mode_split_2, end)
    grasp_sampling_mode: str = "mixed"
    # Split points for grasp_sampling_mode.
    grasp_mode_split_1: int = 200
    grasp_mode_split_2: int = 400
    # Default goal pose mode for assets not explicitly listed in
    # goal_pose_upright_asset_file / goal_pose_lying_asset_file.
    goal_pose_mode: str = "mixed"
    # Legacy selector file: matched assets use goal_pose_mode, unmatched assets stay mixed.
    # Keep this for backward compatibility; do not mix with the two files below.
    goal_pose_asset_file: str = ""
    # New selector files for per-asset goal pose mode control.
    goal_pose_upright_asset_file: str = ""
    goal_pose_lying_asset_file: str = ""

    # goal sampling (env-local)
    goal_x_range: tuple[float, float] = (0.2, 0.8)
    goal_y_range: tuple[float, float] = (-0.4, 0.4)
    # If enabled, sample goal pose orientation from top-k nearest poses to current object orientation.
    goal_pose_orientation_topk_enabled: bool = False
    goal_pose_orientation_topk: int = 15

    # Optional goal pose visualization in viewer/video.
    visualize_goal_pose_marker: bool = False
    goal_pose_marker_max_envs: int = 1
    goal_pose_marker_axis_length: float = 0.08
    goal_pose_marker_line_width: float = 2.0

    # table height sampling (env-local z)
    table_z_range: tuple[float, float] = (-0.06, 0.34)
    # If enabled, shift table_z query when sampling initial arm pose for place reset.
    # Useful when initial-pose data is generated in upright mode with side-approach bias.
    init_arm_upright: bool = False
    init_arm_upright_z_offset: float = 0.0
    # Uniform random jitter applied on top of offset: U(-jitter, +jitter).
    init_arm_upright_z_jitter: float = 0.0
    # If enabled, when reset initialization yields a head-down object (object local +Z points downward),
    # rotate wrist joint by +pi and keep the sampled ee-relative grasp pose unchanged.
    flip_wrist_if_object_upside_down: bool = False
    # Failure when object root z drops below table bottom z by this margin.
    drop_fail_margin: float = 0.01

    # Reward Weights
    w_rew_contact: float = 0.25
    # Near goal (XY+|Z| threshold), contact becomes a penalty to encourage release.
    enable_near_goal_release_contact_penalty: bool = False
    w_rew_contact_near_goal: float = 0.5
    release_near_goal_tol_xy: float = 0.03
    release_near_goal_tol_z: float = 0.015
    w_rew_dist: float = -2.0
    w_rew_object_pos: float = 500.0
    w_rew_object_height: float = 500.0
    w_rew_object_rot: float = 200.0
    w_rew_reached_once: float = 500.0
    w_rew_reached: float = 10.0
    w_rew_reached_release: float = 10.0
    w_rew_success: float = 25.0
    w_rew_action: float = -0.001
    w_rew_failure: float = -1000.0
